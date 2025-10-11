#from os import startfile
from  ast import literal_eval
from pandas import read_csv, read_excel, DataFrame, Series, to_datetime
from numpy import sqrt, array, rint, linspace
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from mplsoccer.pitch import Pitch
from scipy.ndimage import gaussian_filter
from io import BytesIO
import matplotlib.pyplot as plt
import time
from reportlab.graphics import renderPDF
#from svglib.svglib import svg2rlg
from scipy.interpolate import make_interp_spline, BSpline

import matplotlib
##matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import re

from plottable import ColumnDefinition, Table
from plottable.formatters import decimal_to_percent
from plottable.plots import bar, percentile_bars, percentile_stars, progress_donut
from PIL import Image


from joblib import load
import math
from math import sqrt
import streamlit as st

halfs = ['1st Half', '2nd Half']

def createDefensiveMask(data):
    mask = (data['Event']=='Foul Committed') | \
                  (data['Extra 1']=='Tackle')     |  \
                  (data['Event']=='Goal Keeper')  | \
                  (data['Event']=='Interception') |  \
                  (data['Event']=='Duel')         | (data['Event']=='Clearance')      | \
                  (data['Event']=='Block')        | (data['Event']=='Ball Recovery')   | \
                  (data['Event']=='Pressure')     | (data['Event']=='Shield')         | \
                  (data['Event']=='Dribbled Past')| (data['Event']=='Error')
    
    return mask

def createChallengeWonMask(actionData):
    
    """ Challenges Won masks """
    dribbleCompleteMask = (actionData['Event']=='Dribble') & (actionData['Outcome']=='Complete')
    tackleWonMask = (actionData['Extra 1']=='Tackle')  & ((actionData['Outcome']=='Success In Play') | (actionData['Outcome']=='Won')|(actionData['Outcome']=='Success Out'))
    shieldMask = actionData['Event']=='Shield'
    # aerialWonMask = actionData['Extra 3'] =='Aerial Won'
    
    wonMask = dribbleCompleteMask | tackleWonMask | shieldMask #| aerialWonMask
    return wonMask

def createChallengeLostMask(actionData):
    
    """ Challenges Won masks """
    tackleLostMask = (actionData['Extra 1']=='Tackle')  & (((actionData['Outcome']=='Lost Out')|(actionData['Outcome']=='Lost In Play')))
    dispossessedMask = actionData['Event']=='Dispossessed'
    dribbledPastMask = actionData['Event']=='Dribbled Past'
    dribbleInCompleteMask = (actionData['Event']=='Dribble') & (actionData['Outcome']=='Incomplete')
    
    lostMask = tackleLostMask | dispossessedMask | dribbledPastMask | dribbleInCompleteMask
    return lostMask

def createAttackingMask(data):
    attackingMask = (data['Event']=='Ground Pass')  | (data['Event']=='High Pass')      | \
                (data['Event']=='Low Pass')     | (data['Event']=='Dispossessed')   | \
                (data['Event']=='Dribble')        | \
                (data['Event']=='Miscontrol')   |  \
                (data['Event']=='Shot')
    
    return attackingMask
def getKeyPass(data, dataShot):

    keyPassIndex = []
    for i in dataShot['index']:
        for j in range(1, 10):
            prevAction = data.iloc[i-j]
            if((prevAction['Event']=='High Pass') or (prevAction['Event']=='Low Pass') or (prevAction['Event']=='Ground Pass')):
                keyPassIndex.append(i-j)
                break
            elif((prevAction['Event']=='Dribble')):
                pass
            else:
                break
    return data.iloc[keyPassIndex]    

def getAssist(data, dataShot):
    assistIndex = []

    # نخلي dataShot فيها التسديدات اللي جابت جوان بس
    goals = dataShot[dataShot['Outcome'] == 'Goal']

    for i in goals['index']:
        for j in range(1, 10):
            if i - j < 0:  # علشان ما يحصلش IndexError
                break

            prevAction = data.iloc[i - j]

            if prevAction['Event'] in ['High Pass', 'Low Pass', 'Ground Pass']:
                assistIndex.append(i - j)
                break
            elif prevAction['Event'] == 'Dribble':
                continue
            else:
                break

    return data.iloc[assistIndex]

def getSecondAssist(data, dataShot):
    secondAssistIndex = []
    
    # نخلي dataShot فيها التسديدات اللي جابت جوان بس
    goals = dataShot[dataShot['Outcome'] == 'Goal']
    
    for i in goals['index']:
        pass_found = 0  # counter to track passes before goal
        for j in range(1, 20):  # نوسع شوية عشان نلاقي 2 باس
            prevAction = data.iloc[i - j]

            if prevAction['Event'] in ['High Pass', 'Low Pass', 'Ground Pass']:
                pass_found += 1
                if pass_found == 2:   # لو ده تاني باس
                    secondAssistIndex.append(i - j)
                    break
            elif prevAction['Event'] == 'Dribble':
                continue
            else:
                break

    return data.iloc[secondAssistIndex]

def createPassMask(data):
    return (data['Event']=='Ground Pass')  | (data['Event']=='High Pass') | \
                (data['Event']=='Low Pass')

def Calculates_progressive_pass(df):
    """
    Calculates progressive passes.

    Parameters:
    df (DataFrame): A pandas DataFrame containing the event data with columns:
        - 'Event': The type of action (contains 'Pass')
        - 'Actions positions x', 'Actions positions y': Starting coordinates of the action
        - 'End Location x End', 'End Location y End': Ending coordinates of the action

    Returns:
    DataFrame: The input DataFrame with a new column:
        - 'prog_pass': True if the pass meets the conditions for being progressive, otherwise False
    """

    # Initialize 'prog_pass' to False
    df['prog_pass'] = False

    # Determine if the pass is progressive based on conditions
    df.loc[
        (df['Actions positions x'] <= 60) & 
        (df['Actions positions x End'] <= 60) & 
        (df['Actions Pos Length'] >= 35), 'prog_pass'] = True

    # Check second condition
    df.loc[
        (df['Actions positions x'] <= 60) & 
        (df['Actions positions x End'] > 60) & 
        (df['Actions Pos Length'] >= 15), 'prog_pass'] = True

    # Check third condition
    df.loc[
        (df['Actions positions x'] >= 60) & 
        (df['Actions positions x End'] >= 60) & 
        (df['Actions Pos Length'] >= 10), 'prog_pass'] = True


    return df

def process_and_predict_xG(data, model_path='IQStats_Xg.joblib'):
    """
    دالة لمعالجة بيانات التسديدات، حساب الميزات، وتوقع احتمالية تسجيل الهدف (xG).
    
    المدخلات:
    - data: DataFrame يحتوي على بيانات المباراة.
    - model_path: مسار النموذج المحفوظ بصيغة joblib.
    
    المخرجات:
    - DataFrame يحتوي على العمود الجديد IQStats_xG.
    """
    # تحميل النموذج
    IQStats_model = load(model_path)

    # التحقق من الأعمدة المطلوبة
    required_columns = ['Event', 'Actions positions x', 'Actions positions y', 'Extra 1', 'Extra 2', 'Extra 4']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # معالجة بيانات التسديدات
    data = data.fillna({'Extra 1': '', 'Extra 2': '', 'Extra 4': ''})
    df_shot = data.loc[data['Event'] == 'Shot'].copy()

    # إنشاء الأعمدة الأساسية
    df_shot['x'] = df_shot['Actions positions x']
    df_shot['y'] = df_shot['Actions positions y']
    df_shot['body_part_name'] = df_shot['Extra 2']

    df_shot['under_pressure'] = np.where(df_shot['Extra 4'] == 'under_pressure', 1, 0)
    df_shot['header'] = np.where(df_shot['Extra 2'] == 'Head', 1, 0)
    df_shot['technique_name_Backheel'] = np.where(df_shot['Extra 4'] == 'Backheel', 1, 0)
    df_shot['technique_name_Diving Header'] = np.where(df_shot['Extra 4'] == 'Diving Header', 1, 0)
    df_shot['technique_name_Half Volley'] = np.where(df_shot['Extra 4'] == 'Half Volley', 1, 0)
    df_shot['technique_name_Lob'] = np.where(df_shot['Extra 4'] == 'Lob', 1, 0)
    df_shot['technique_name_Normal'] = np.where(df_shot['Extra 4'] == 'Normal', 1, 0)
    df_shot['technique_name_Overhead Kick'] = np.where(df_shot['Extra 4'] == 'Overhead Kick', 1, 0)
    df_shot['technique_name_Volley'] = np.where(df_shot['Extra 4'] == 'Volley', 1, 0)
    df_shot['sub_type_name_Corner'] = np.where(df_shot['Extra 1'] == 'Corner', 1, 0)
    df_shot['sub_type_name_Free Kick'] = np.where(df_shot['Extra 1'] == 'Free Kick', 1, 0)
    df_shot['sub_type_name_Open Play'] = np.where(df_shot['Extra 1'] == 'Open Play', 1, 0)
    df_shot['sub_type_name_Penalty'] = np.where(df_shot['Extra 1'] == 'Penalty', 1, 0)

    # حساب الميزات الإضافية
    def calculate_angle(x, y):
        g0 = [120, 44]  # القائم الأعلى
        g1 = [120, 36]  # القائم السفلي
        p = [x, y]

        v0 = np.array(g0) - np.array(p)
        v1 = np.array(g1) - np.array(p)

        angle = math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return abs(np.degrees(angle))

    def calculate_distance(x, y):
        x_dist = 120 - x
        if y < 36:
            y_dist = 36 - y
        elif y > 44:
            y_dist = y - 44
        else:
            y_dist = 0
        return math.sqrt(x_dist*2 + y_dist*2)

    def is_preferable_side(y, body_part_name):
        if y < 40:
            side = 'left'
        elif y > 40:
            side = 'right'
        else:
            side = 'center'

        if ((side == 'left') and (body_part_name == 'Right Foot')) or \
           ((side == 'right') and (body_part_name == 'Left Foot')):
            return 1
        return 0

    df_shot['angle'] = df_shot.apply(lambda row: calculate_angle(row['x'], row['y']), axis=1)
    df_shot['distance'] = df_shot.apply(lambda row: calculate_distance(row['x'], row['y']), axis=1)
    df_shot['preferable_side'] = df_shot.apply(lambda row: is_preferable_side(row['y'], row['body_part_name']), axis=1)

    # ملء القيم الناقصة
    df_shot['under_pressure'] = df_shot['under_pressure'].fillna(0).astype(int)

    # تحديد الأعمدة المستخدمة في التنبؤ
    X_cols = ['under_pressure', 'angle', 'distance',
              'preferable_side', 'header', 'technique_name_Backheel',
              'technique_name_Diving Header', 'technique_name_Half Volley',
              'technique_name_Lob', 'technique_name_Normal',
              'technique_name_Overhead Kick', 'technique_name_Volley',
              'sub_type_name_Corner', 'sub_type_name_Free Kick',
              'sub_type_name_Open Play', 'sub_type_name_Penalty']

    df_shot = df_shot.dropna(subset=X_cols)
    X = df_shot[X_cols].dropna()

    # توقع xG باستخدام النموذج
    df_shot['xG'] = IQStats_model.predict_proba(X)[:, 1]

    # دمج عمود IQStats_xG مع البيانات الأصلية
    data = data.merge(df_shot[['xG']], left_index=True, right_index=True, how='left')
    
    return data


####### start def 
def data_pre_procces(actionsFileDir,start_side):
    
    actionsFileDir.seek(0)
    data = pd.read_csv(actionsFileDir, delimiter='\t', encoding="utf-16") 
    data['Position'] = to_datetime(data['Position']) 
    # تحويل وقت الفيديو إلى دقائق
    data['video_minute'] = (
        data['Position'].dt.hour * 60 +
        data['Position'].dt.minute +
        data['Position'].dt.second / 60
    )
    # نحدد بداية كل شوط
    first_half_start = data.loc[data['Half'] == '1st Half', 'video_minute'].min()
    second_half_start = data.loc[data['Half'] == '2nd Half', 'video_minute'].min()
    # نطرح البداية ونعدل الوقت
    data.loc[data['Half'] == '1st Half', 'match_minute'] = data['video_minute'] - first_half_start
    data.loc[data['Half'] == '2nd Half', 'match_minute'] = 45 + (data['video_minute'] - second_half_start)
    
    playersNames = Series(data['Player 1'].unique()).dropna().reset_index(drop=True)
    """ x , y for  Data """
    data['Actions positions'] = (data['Start Location'].combine_first(data['Field Position']))
    # data = data.dropna(subset=['Actions positions'])
    data['Actions positions x'] = data['Actions positions'][:].str.split(";", n = 1, expand = True)[0].astype(float)
    data['Actions positions y'] = data['Actions positions'][:].str.split(";", n = 1, expand = True)[1].astype(float)
    
    data['Actions positions x End'] = data['End Location'][:].str.split(";", n = 1, expand = True)[0].astype(float)
    data['Actions positions y End'] = data['End Location'][:].str.split(";", n = 1, expand = True)[1].astype(float)
    
    
    data = data.reset_index()
        # تعريف الشوطين
   # halfs = ['1st Half', '2nd Half']
    
    #if(data[data['Event']=='Goal Keeper']['Actions positions x'].iloc[0]<60):
    #    halfsToChangeXY = [halfs[1],halfs[0]]
    #else:
    #    halfsToChangeXY = [halfs[0],halfs[1]]
    #""" change x and y position """
    #temp = data[(data['Half']==halfsToChangeXY[0])]
    #idxS = temp.head(1).index[0]
    #idxE = temp.tail(1).index[0]
    #for i in range(idxS, idxE+1):
    #    data.loc[i, ['Actions positions x']] = 120-data.loc[i, ['Actions positions x']]
    #    data.loc[i, ['Actions positions y']] = 80 -data.loc[i, ['Actions positions y']]
    #    data.loc[i, ['Actions positions x End']] = 120-data.loc[i, ['Actions positions x End']]
    #    data.loc[i, ['Actions positions y End']] = 80 -data.loc[i, ['Actions positions y End']]

    
    # 🧭 الاتجاه اليدوي في حالة عدم وجود حارس
    # يمكن أن يكون "left" أو "right"
    start_side = start_side  # ← غيّر هنا حسب الفريق بدأ منين

    halfs = ['1st Half', '2nd Half']

    # ==================================================
    # 1️⃣ تحديد الاتجاه من الحارس (لو موجود)
    # ==================================================
    goalkeeper_events = data[data['Event'] == 'Goal Keeper']

    if not goalkeeper_events.empty:
        first_gk_x = goalkeeper_events['Actions positions x'].iloc[0]
        if first_gk_x < 60:
            # بدأ من اليسار → نقلب الشوط الثاني
            halfsToChangeXY = [halfs[1]]
            print("✅ الفريق بدأ من اليسار — سيتم قلب الشوط الثاني.")
        else:
            # بدأ من اليمين → نقلب الشوط الأول
            halfsToChangeXY = [halfs[0]]
            print("✅ الفريق بدأ من اليمين — سيتم قلب الشوط الأول.")
    else:
        # ==================================================
        # 2️⃣ الاتجاه اليدوي كخطة بديلة
        # ==================================================
        print("⚠️ لا يوجد حدث لحارس المرمى، سيتم استخدام الاتجاه اليدوي.")

        if start_side.lower() == "left":
            # نفس منطق الحارس: لو بدأ من اليسار → نقلب الشوط الثاني
            halfsToChangeXY = [halfs[1]]
            print("➡️ الفريق بدأ من اليسار — سيتم قلب الشوط الثاني.")
        elif start_side.lower() == "right":
            # لو بدأ من اليمين → نقلب الشوط الأول
            halfsToChangeXY = [halfs[0]]
            print("➡️ الفريق بدأ من اليمين — سيتم قلب الشوط الأول.")
        else:
            halfsToChangeXY = [halfs[1]]
            print("⚠️ اتجاه غير معروف، تم استخدام الافتراضي (left): سيتم قلب الشوط الثاني.")

    # ==================================================
    # 3️⃣ قلب الاتجاهات باستخدام apply()
    # ==================================================
    def flip_coordinates(row):
        if row['Half'] in halfsToChangeXY:
            row['Actions positions x'] = 120 - row['Actions positions x']
            row['Actions positions y'] = 80 - row['Actions positions y']
            row['Actions positions x End'] = 120 - row['Actions positions x End']
            row['Actions positions y End'] = 80 - row['Actions positions y End']
        return row

    data = data.apply(flip_coordinates, axis=1)

    
    data['Actions Pos Length'] = np.sqrt(
        (data['Actions positions x End'] - data['Actions positions x'])**2 +
        (data['Actions positions y End'] - data['Actions positions y'])**2
    )
    
    
    data = Calculates_progressive_pass(data)
    data = process_and_predict_xG(data, model_path='IQStats_Xg.joblib')
    
    actionsMask =   (data['Event']=='Ground Pass')  | (data['Event']=='High Pass')      | \
                    (data['Event']=='Low Pass')     | (data['Event']=='Dispossessed')   | \
                    (data['Extra 1']=='Aerial Lost')| (data['Event']=='Foul Committed') | \
                    (data['Extra 1']=='Tackle')     | (data['Event']=='Dribble')        | \
                    (data['Event']=='Miscontrol')   | (data['Event']=='Goal Keeper')    | \
                    (data['Event']=='Interception') | (data['Event']=='Shot')           | \
                    (data['Event']=='Duel')         | (data['Event']=='Clearance')      | \
                    (data['Event']=='Block')        |(data['Event']=='Ball Recovery')   | \
                    (data['Event']=='Pressure')     | (data['Event']=='Shield')         | \
                    (data['Event']=='Dribbled Past')| (data['Event']=='Error')
        
    actionData = data[actionsMask].reset_index()
    
    
    badHighPassMask         = (actionData['Event']   =='High Pass')   &((actionData['Outcome']=='Incomplete') | (actionData['Outcome']=='Out') | (actionData['Outcome']=='Pass Offside') | (actionData['Outcome']=='Unknown'))
    badGroundPassMask       = (actionData['Event']   =='Ground Pass') &((actionData['Outcome']=='Incomplete') | (actionData['Outcome']=='Out') | (actionData['Outcome']=='Pass Offside') | (actionData['Outcome']=='Unknown'))
    badLowPassMask          = (actionData['Event']   =='Low Pass')    &((actionData['Outcome']=='Incomplete') | (actionData['Outcome']=='Out') | (actionData['Outcome']=='Pass Offside') | (actionData['Outcome']=='Unknown'))
    badDispossessedMask     =  actionData['Event']   =='Dispossessed'
    badAerialLostMask       =  actionData['Extra 1'] =='Aerial Lost'
    badFoulCommittedMask    =  actionData['Event']   =='Foul Committed'
    badTackleMask           = (actionData['Extra 1'] =='Tackle')  & (actionData['Outcome']=='Lost Out')
    badDribbleMask          = (actionData['Event']   =='Dribble') & (actionData['Outcome']=='Incomplete')
    badMiscontrolMask       =  actionData['Event']   =='Miscontrol'
    badGoalKeeperMask       = (actionData['Event']   =='Goal Keeper') &(actionData['Extra 1'] =='Goal Conceded')
    badErrorMask            =  actionData['Event']   =='Error'
    badInterceptionMask     = (actionData['Event'] =='Interception')  & ((actionData['Outcome']=='Lost Out') | (actionData['Outcome']=='Lost In Play'))
    badRecoveryFailureMask  =  actionData['Outcome']   =='Recovery Failure'
    badShotMask             = (actionData['Event']   =='Shot')   &((actionData['Outcome']=='Blocked') | (actionData['Outcome']=='Wayward') | (actionData['Outcome']=='Off T'))
    
    actionBadMask = badHighPassMask     |\
                badGroundPassMask       |\
                badLowPassMask          |\
                badDispossessedMask     |\
                badAerialLostMask       |\
                badFoulCommittedMask    |\
                badTackleMask           |\
                badDribbleMask          |\
                badMiscontrolMask       |\
                badGoalKeeperMask       |\
                badErrorMask            |\
                badInterceptionMask     |\
                badRecoveryFailureMask  |\
                badShotMask             
    
    actionBadData = actionData[actionBadMask]
    actionGoodData = actionData[~actionBadMask]
    
    dataShot = data[data['Event']=='Shot']
    
    badShotMask = (dataShot['Event']   =='Shot')   &((dataShot['Outcome']=='Blocked') | (dataShot['Outcome']=='Wayward') | (dataShot['Outcome']=='Off T'))
    
    goodShotMask = ~badShotMask
    
    dataShotGood = dataShot[goodShotMask]
    
    dataShotBad = dataShot[badShotMask]
    
    wonMask = createChallengeWonMask(data)
    lostMask = createChallengeLostMask(data)
    
    
    dataChallengeWon = data[wonMask]
    dataChallengeLost = data[lostMask]
    
    challengeMask = wonMask | lostMask
    dataChallenge = data[challengeMask]
    
    
    maskPass =  createPassMask(data)
    dataPass = data[maskPass]
    dataprogpass = dataPass[dataPass['prog_pass']==True]
    
    maskPassGood = createPassMask(actionGoodData)
    dataPassGood = actionGoodData[maskPassGood]
    dataprogpassGood = dataPassGood[dataPassGood['prog_pass']==True]
    
    maskPassBad = createPassMask(actionBadData)
    dataPassBad = actionBadData[maskPassBad]
    dataprogpassBad = dataPassBad[dataPassBad['prog_pass']==True]
    
    attackingChallengeMask = createAttackingMask(dataChallenge)
    
    
    defensiveChallengeMask = createDefensiveMask(dataChallenge)
    
    attackingChallengeMaskWon = createAttackingMask(dataChallengeWon)
    
    defensiveChallengeMaskWon = createDefensiveMask(dataChallengeWon)
    
    dataChallengeDefensive = dataChallenge[defensiveChallengeMask]
    
    dataChallengeDefensiveWon = dataChallengeWon[defensiveChallengeMaskWon]
    
    dataChallengeAttacking = dataChallenge[attackingChallengeMask]
    dataChallengeAttackingWon = dataChallengeWon[attackingChallengeMaskWon]
    
    
    dataDribble = dataChallenge[dataChallenge['Event']=='Dribble']
    dataDribbleWon = dataChallengeWon[dataChallengeWon['Event']=='Dribble']
    dataDribbleLost = dataChallengeLost[dataChallengeLost['Event']=='Dribble']
    
    dataTackle = dataChallenge[dataChallenge['Extra 1']=='Tackle']
    dataTackleWon = dataChallengeWon[dataChallengeWon['Extra 1']=='Tackle']
    dataTackleLost = dataChallengeLost[dataChallengeLost['Extra 1']=='Tackle']
    
    
    interceptionData = data[data['Event']=='Interception']
    interceptionDataWon = interceptionData[interceptionData['Actions positions x']>=60]
    
    dataFoulsWon = data[data['Event']=='Foul Won']
    dataFoulsCommitted = data[data['Event']=='Foul Committed']
    
    dataOffside = data[data['Outcome']=='Pass Offside']
    
    shotInBoxMask = (dataShot['Actions positions x']>=103.5)&(dataShot['Actions positions y']>=19.9)&(dataShot['Actions positions y']<=80-19.9)
    
    shotInBoxMaskGood = (dataShotGood['Actions positions x']>=103.5)&(dataShotGood['Actions positions y']>=19.9)&(dataShotGood['Actions positions y']<=80-19.9)
    
    aerialMask = (data['Extra 1']=='Aerial Lost') |(data['Extra 3']=='Aerial Won')
    dataAerial = data[aerialMask]
    
    aerialMaskWon = data['Extra 3']=='Aerial Won'
    dataAerialWon = data[aerialMaskWon]
    
    aerialMaskLost = data['Extra 1']=='Aerial Lost'
    dataAerialLost = data[aerialMaskLost]
    
    dataKeyPass = getKeyPass(data, dataShot)
    dataKeyPassGood = getKeyPass(data, dataShotGood)
    dataKeyPassBad = getKeyPass(data, dataShotBad)
    
    dataAsist = getAssist(data, dataShot)
    dataSecondAsist = getSecondAssist(data, dataShot)
    
    dataPressure = data[data['Event']=='Pressure']
    
    
    dataDefensiveMask = createDefensiveMask(actionData)
    
    dataDefensive = actionData[dataDefensiveMask]
    
    dataAttackingMask = createAttackingMask(actionData)
    
    dataAttacking = actionData[dataAttackingMask]
    
    
    """ Lost balls. Recovered balls  """
    dataRecoveredBall = data[data['Event']=='Ball Recovery']
    
    badHighPassMask         = (data['Event']   =='High Pass')   &((data['Outcome']=='Incomplete') | (data['Outcome']=='Out') | (data['Outcome']=='Pass Offside') | (data['Outcome']=='Unknown'))
    badGroundPassMask       = (data['Event']   =='Ground Pass') &((data['Outcome']=='Incomplete') | (data['Outcome']=='Out') | (data['Outcome']=='Pass Offside') | (data['Outcome']=='Unknown'))
    badLowPassMask          = (data['Event']   =='Low Pass')    &((data['Outcome']=='Incomplete') | (data['Outcome']=='Out') | (data['Outcome']=='Pass Offside') | (data['Outcome']=='Unknown'))
    
    lostBallMask = badHighPassMask | badGroundPassMask | badLowPassMask
    dataLostBall = data[lostBallMask]
    
    
    dataClearance = data[data['Event']=='Clearance']
    dataCross = dataPass[dataPass['Extra 3']=='Cross']
    dataCrossGood = dataPassGood[dataPassGood['Extra 3']=='Cross']
    
    dataThroughBall = dataPass[dataPass['Extra 6']=='Through Ball']
    dataError =  data[data['Event']=='Error']
    return (
        data,
        actionData,
        actionBadData,
        actionGoodData,
        dataShot,
        dataShotGood,
        dataShotBad,
        dataChallengeWon,
        dataChallengeLost,
        dataChallenge,
        dataPass,
        dataprogpass,
        dataPassGood,
        dataprogpassGood,
        dataPassBad,
        dataprogpassBad,
        dataChallengeDefensive,
        dataChallengeDefensiveWon,
        dataChallengeAttacking,
        dataChallengeAttackingWon,
        dataDribble,
        dataDribbleWon,
        dataDribbleLost,
        dataTackle,
        dataTackleWon,
        dataTackleLost,
        interceptionData,
        interceptionDataWon,
        dataFoulsWon,
        dataFoulsCommitted,
        dataOffside,
        dataAerial,
        dataAerialWon,
        dataAerialLost,
        dataKeyPass,
        dataKeyPassGood,
        dataKeyPassBad,
        dataAsist,
        dataSecondAsist,
        dataPressure,
        dataDefensive,
        dataAttacking,
        dataRecoveredBall,
        dataLostBall,
        dataClearance,
        dataCross,
        dataCrossGood,
        dataThroughBall,
        dataError
    )

