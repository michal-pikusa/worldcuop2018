import pandas as pd
import numpy as np
import xgboost as xgb
import itertools
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def make_train_dataset():
    # load datasets
    stats = pd.read_csv('team_stats.csv')
    results = pd.read_csv('match_results.csv')
    # average the stats for a given year
    stats = stats.groupby(['Name','Year'], as_index=False, sort=False)[['ATT','MID','DEF','OVR']].mean()
    # add stats to the home team
    results = results.merge(stats,  how='left', left_on=['home_team','Year'], right_on = ['Name','Year'])
    # add stats to the away team
    results = results.merge(stats,  how='left', left_on=['away_team','Year'], right_on = ['Name','Year'])
    results = results.dropna()
    # calculate scores, filter out draws, and indicate a winner
    results['score'] = results['home_score'] - results['away_score']
    results = results[results['score'] != 0]
    results['winner'] = 0
    results['winner'][results['score'] > 0] = 1
    labels = results['winner'].tolist()
    # filter to important values only
    results = results[['ATT_x','MID_x','DEF_x','OVR_x','ATT_y','MID_y','DEF_y','OVR_y','Year']]
    # return results and the labels as numpy arrays
    results = results.as_matrix()
    return results,np.array(labels)

def make_match_dataset(team1,team2):
    # load world cup stats
    wc_stats = pd.read_csv('wc2018.csv')
    # extract stats for both teams 
    team1_values = wc_stats[wc_stats['Name'] == team1].values.tolist()[0][1:-1]
    team2_values = wc_stats[wc_stats['Name'] == team2].values.tolist()[0][1:-1]
    values = team1_values + team2_values
    # add randomness (football is unpredictable!)
    values = [np.random.normal(x, scale=0.5) for x in values]
    values.append(2018)
    return np.array(values)

def train_model(train_dataset,train_labels,params):
    # convert the dataset into xgb object
    dtrain = xgb.DMatrix(train_dataset, label=train_labels)
    n_iter = 20  # the number of training iterations
    # train the model
    model = xgb.train(params, dtrain, n_iter)
    return model

def evaluate_model(train_dataset,train_labels,params):
    # split the dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(train_dataset, train_labels, test_size=0.2, random_state=42)
    # convert to xgb objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    n_iter = 20
    # train the model
    model = xgb.train(params, dtrain, n_iter)
    # predict the probabilities on the test set and assign classes
    preds = model.predict(dtest)
    pred_class = [1 if x > 0.5 else 0 for x in preds]
    # calculate and print accuracy
    acc = accuracy_score(y_test, pred_class)
    print("Accuracy: "+str(acc))

def predict_match_outcome(team1,team2,model,n_sim):
    # make the dataset out of stats from two teams
    match_dataset = make_match_dataset(team1,team2)
    # reshape (it has to be 2-dimensional) and convert to xgb object
    match_dataset = xgb.DMatrix(match_dataset.reshape(1,-1))
    pred_list = []
    # simulate a number of games specified in n_sim and make a list of predictions
    for i in range(1,n_sim):
        pred = model.predict(match_dataset)
        pred_list.append(pred)
    # average the list of predictions to get the final prediction and assign the winner
    average_pred = np.average(np.array(pred_list))
    if average_pred > 0.5:
        winner = team1
    else:
        winner = team2
        average_pred = 1-average_pred
    return winner,average_pred

if __name__ == '__main__':
    train_dataset, train_labels = make_train_dataset()
    params = {
    'max_depth': 4,  # the maximum depth of each tree
    'eta': 0.1, # the training step for each iteration
    'objective': 'binary:logistic', # error evaluation for binary classes with probability
    'silent': 1}
    # first evaluate the accuracy of the model on given parameters
    evaluate_model(train_dataset,train_labels,params)
    # train the model on the whole dataset
    model = train_model(train_dataset,train_labels,params)
    team_list = ['Uruguay','France','Brazil','Belgium','Sweden','England','Russia','Croatia']
    # pair up the teams and simulate the games using 1000 simulations for each game
    for a, b in itertools.combinations(team_list, 2):
        winner,average_pred = predict_match_outcome(a,b,model,1000)
        print(str(a)+' vs. '+str(b))
        print('Winner: '+str(winner)+' Probability: '+str(average_pred))
