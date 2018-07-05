import pandas as pd
import numpy as np
import xgboost as xgb
import itertools

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

def train_model(train_dataset,train_labels):
    dtrain = xgb.DMatrix(train_dataset, label=train_labels)
    params = {
    'max_depth': 7,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'objective': 'binary:logistic'}  # error evaluation for binary classes
    n_iter = 20  # the number of training iterations
    model = xgb.train(params, dtrain, n_iter)
    return model

def predict_match_outcome(team1,team2,model,n_sim):
    match_dataset = make_match_dataset(team1,team2)
    match_dataset = xgb.DMatrix(match_dataset.reshape(1,-1))
    pred_list = []
    for i in range(1,n_sim):
        pred = model.predict(match_dataset)
        pred_list.append(pred)
    average_pred = np.average(np.array(pred_list))
    if average_pred > 0.5:
        winner = team1
    else:
        winner = team2
        average_pred = 1-average_pred
    return winner,average_pred

if __name__ == '__main__':
    train_dataset, train_labels = make_train_dataset()
    model = train_model(train_dataset,train_labels)
    team_list = ['Uruguay','France','Brazil','Belgium','Sweden','England','Russia','Croatia']
    for a, b in itertools.combinations(team_list, 2):
        winner,average_pred = predict_match_outcome(a,b,model,1000)
        print(str(a)+' vs. '+str(b))
        print('Winner: '+str(winner)+' Probability: '+str(average_pred))