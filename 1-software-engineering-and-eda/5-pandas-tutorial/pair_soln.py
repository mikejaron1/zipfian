import pandas as pd

######################################################
########## PART 1: FEATURE ENGENIEERING #############
######################################################

def format_batting_data():
    dfBatting = pd.read_csv('data/baseball-csvs/Batting.csv')

    # batting average
    # BA = HITS / AT BATS
    dfBatting['BA'] =  dfBatting['H'] / dfBatting['AB'] 


    ### ON BASE PERCENTAGE
    # OBP = HITS + BASE BY BALLS + HIT BY PITCH /  (AT BATS + BASE BY BALLS + HIT BY PITCH + SACRAFICED FLY )
    # OBP = H + BB + HBP / AB + BB + HBP + SF
    dfBatting['OBP'] = (dfBatting['H'] + dfBatting['BB'] + dfBatting['HBP']) /  (dfBatting['AB'] + dfBatting['BB'] + dfBatting['HBP'] + dfBatting['SF'])


    ### SLUGGING AVERAGE SLG
    # before we do this wee need to calculate singles, which hopefull is jsut hits - sum(2b, 3b, hr)
    dfBatting['1B'] = dfBatting['H'] - (dfBatting['2B'] + dfBatting['3B'] + dfBatting['HR'] )


    dfBatting['SLG'] = ( (1 * dfBatting['1B']) + (2 * dfBatting['2B']) + (3 * dfBatting['3B']) + (4 * dfBatting['HR']) ) / dfBatting['AB']
    return dfBatting



######################################################
########## WORKING WITH SALARIES.CSV     #############
######################################################

def format_salary_data():
    dfSals = pd.read_csv('data/baseball-csvs/Salaries.csv')
    return dfSals




######################################################
######  MERGING THE SALARY AND BATTING DATA ##########
######################################################

def merge_batting_and_sals():
    dfSals = format_salary_data()
    dfBatting = format_batting_data()

    ### REMOVE ALL THE DATA THAT IS BELOW 1985
    dfBatting = dfBatting[dfBatting['yearID'] >= 1985]

    ### MERGE THE TWO TOGETHER ON A DOUBLE CONDITION
    mergeddf = pd.merge(dfBatting, dfSals, on=['playerID', 'yearID'])

    ### THIS IS A WAY TO DROP A COLUMN IN-PLACE
    del mergeddf['G_old']

    ### THIS IS A WAY TO DROP MULTIPLE COLUMNS (NOTE THE AXIS=1, IF YOU DO AXIS=0 IT WILL DROP ROWS)
    mergeddf =  mergeddf.drop(['teamID_y', 'lgID_y'], axis=1)
    return mergeddf



######################################################
######  SEARCHING FOR THE RIGHT PLAYERS     ##########
######################################################


def main():
    mergeddf = merge_batting_and_sals()

    
    condition1 = mergeddf.yearID == 2001
    condition2 = mergeddf.teamID_x == 'OAK'
    oak2001 = mergeddf[ condition1 & condition2]

    ### FIND THE STATS FOR THE PLAYERS WE ARE MISSING
    # THIS IS A LIST OF THE PLAYERS WE ARE LOSING
    lostboys = ['isrinja01', 'giambja01', 'damonjo01', 'saenzol01']

    # CREATE A BOOLEAN MASK THAT RETURNS TRUE OR FALSE REGARDING IF THE ELEMENTS IN OUR LIST ARE IN THE PLAYERID COLUMN
    mask = oak2001['playerID'].isin(lostboys)

    # USING OUR MASK, COPY JUST THE TRUE STATEMTNTS DATA INTO A NEW DATA FRAOM CALLED LOSTBOYSDF
    lostboysdf = oak2001[mask]

    ### CREATE A CONDITION IN WHICH ONLY YEILDS THE DATA WHEN THE COLUMN YEARID IS 2001
    condition3 = mergeddf.yearID == 2001

    ### APPLY THAT CONDITION TO mergeddf AND SET IT EQUAL TO A NEW DATAFRAME
    all2001 = mergeddf[condition3]

    ### CREATE ANOTHER CONDITION, THAT YEILDS ONLY THE DATA WHEN THE COLUMN 'AB' IS ABOVE 40
    condition4 = all2001.AB >= 40

    ### APPLY THAT CONDITION, AND SET IT EQUAL TO ITS SELF (THUS OVERRIDING IT)
    all2001 = all2001[condition4]

    ### SELECT ONLY THE COLUMNS WE CARE ABOUT, AND SET IT EQUAL TO ITSELF (THUS OVERRIDING IT)

    all2001 = all2001[['playerID', 'teamID_x','AB','HR','SLG', 'OBP', 'salary']]
    # all2001 = all2001.sort('OBP', ascending=False).sort('salary', ascending=True)

    ### SORT BY OPB, IN DESCENDING ORDER
    all2001 = all2001.sort('OBP', ascending=False)

    ### CREATE ANOTHER CONDITION THAT ONLY RETURNS PLAYERS LESS THAN 8MILL
    c4 = all2001.salary < 8000000

    ###  '~' does a select inverse.  so instead of returning .isin()
    ###  it will return .isNOTin().  
    c5 = ~all2001['playerID'].isin(lostboys)

    
    answerdf = all2001[c4 & c5]
    print answerdf.head()



if __name__ == '__main__':  # If we're running as a script...
    main()
