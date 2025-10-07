import streamlit as st
import pandas as pd
import numpy as np
from pandas import Series
from datetime import datetime
##from os import startfile
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
#matplotlib.use("TkAgg")
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
from matplotlib.lines import Line2D

from Data_process import (
    createDefensiveMask,
    createChallengeWonMask,
    createChallengeLostMask,
    createAttackingMask,
    getKeyPass,
    getAssist,
    getSecondAssist,
    createPassMask,
    Calculates_progressive_pass,
    process_and_predict_xG
)

# ================= Streamlit Config =================
st.set_page_config(page_title="Dynamic Player Analysis", layout="wide")
st.title("📊 رفع TSV وتحليلات اللاعبين ديناميكية")

# ================= File Upload =================
uploaded_file = st.file_uploader("اختر ملف TSV (UTF-16)", type=["csv"])
if uploaded_file:
    try:
        df_ = pd.read_csv(uploaded_file, sep='\t', encoding='utf-16')
        if df_.empty:
            st.error("الملف فارغ")
            st.stop()
        st.success("تم تحميل الملف بنجاح")
    except Exception as e:
        st.error(f"فشل تحميل الملف: {e}")
        st.stop()

    # ================= Basic Processing =================
    st.dataframe(df_.head())
    from Data_process import data_pre_procces
    (
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
       dataThroughBall
    )= data_pre_procces(uploaded_file)
    st.dataframe(data.head())
# ================== Player Avg Positions & Total Actions =================

    playersNames = sorted(data["Player 1"].dropna().unique())
    playersNames = pd.Series(playersNames) 
    halfs = ['1st Half', '2nd Half']


    
    xAvgList = []
    yAvgList = []
    playerNumberOfActionsTotal = []
    
    for playerName in playersNames:
        playerNumberOfActions = []
        playerData = actionData[actionData['Player 1'] == playerName]
        
        # Actions per 15-min period
        for timePeriod in [0, 15, 30, 45, 60, 75]:
            if timePeriod == 75:
                mask = playerData['match_minute'].between(timePeriod, 200)
            else:
                mask = playerData['match_minute'].between(timePeriod, timePeriod+14)
            playerNumberOfActions.append(playerData['match_minute'][mask].shape[0])
        
        playerNumberOfActionsTotal.append(sum(playerNumberOfActions))
        xAvgList.append(playerData['Actions positions x'].mean())
        yAvgList.append(playerData['Actions positions y'].mean())
    
    # Team actions per 15-min period
    teamNumberOfActions = []
    for timePeriod in [0, 15, 30, 45, 60, 75]:
        if timePeriod == 75:
            mask = actionData['match_minute'].between(timePeriod, 200)
        else:
            mask = actionData['match_minute'].between(timePeriod, timePeriod+14)
        teamNumberOfActions.append(actionData['match_minute'][mask].shape[0]/11)
    
    # Extract player numbers from names (assuming "Number, Name" format)
    playersNumbers = playersNames.str.split(',', n=1, expand=True)[0]

    import pandas as pd
    from pandas import DataFrame
    
    def printRawInTable(nSpaces, actions, actionsGood, rowName):
        nuActions = []
        nuActionsGood = []
        values = []   # العمود اللي فيه العدد / الناجح
        percents = [] # العمود اللي فيه النسبة %
    
        # حساب الأعداد
        for action in actions:
            nuActions.append(action.shape[0])
        
        for actionGood in actionsGood:
            if isinstance(actionGood, DataFrame):
                nuActionsGood.append(actionGood.shape[0])
            else:
                nuActionsGood.append('-')
        
        # بناء النتائج
        for nu, nuGood in zip(nuActions, nuActionsGood):
            if nu > 0:
                if nuGood != '-':
                    values.append(f"{nu} / {nuGood}")
                    percents.append(f"{int(round((nuGood) / (nu) * 100))}%")
                else:
                    values.append(str(nu))
                    percents.append('-')
            else:
                values.append('-')
                percents.append('-')
        
        # إرجاع DataFrame مع عمود إضافي للنسبة
        return DataFrame({
            "PATASTATS INDEX": [rowName],
            "Per Match": [values[0] if len(values) > 0 else '-'],
            "Per Match %": [percents[0] if len(percents) > 0 else '-'],
            "1st half": [values[1] if len(values) > 1 else '-'],
            "1st half %": [percents[1] if len(percents) > 1 else '-'],
            "2nd half": [values[2] if len(values) > 2 else '-'],
            "2nd half %": [percents[2] if len(percents) > 2 else '-']
        })
    
    import pandas as pd
    from pandas import DataFrame
    def addTableRow(dataTable, actions1, actions1Good, rowName, nSpaces=12 ,halfs = ['1st Half', '2nd Half']):
        

        """ Shots Both Halfs """
        actions2 = actions1[actions1['Half']==halfs[0]]
        actions3 = actions1[actions1['Half']==halfs[1]]
        Actions = [actions1, actions2, actions3]
    
        if not isinstance(actions1Good, DataFrame):
            ActionsGood = ['-']*3
        else:
            actions2Good = actions1Good[actions1Good['Half']==halfs[0]]
            actions3Good = actions1Good[actions1Good['Half']==halfs[1]]
            ActionsGood = [actions1Good, actions2Good, actions3Good]
    
        # هنا بقى: printRawInTable بيرجع DataFrame جاهزة
        row_df = printRawInTable(nSpaces, Actions, ActionsGood, rowName)
        
        # نضيفها عالجدول الأساسي
        dataTable = pd.concat([dataTable, row_df], ignore_index=True)
        return dataTable     

    # ================== Select Player =================
    selected_player = st.selectbox("اختر لاعب:", playersNames)
    
    player_data = data[data['Player 1']==selected_player]
    
    st.subheader(f"📋 بيانات اللاعب   {selected_player}")
    st.dataframe(player_data.head())

    playerName = selected_player
    # ================== Example Analyses =================
    # افترض أن playerName مأخوذ من اختيار المستخدم
    playerNumberOfActions = []
    playerData = actionData[actionData['Player 1'] == playerName]
    playerActionBadData = actionBadData[actionBadData['Player 1'] == playerName]
    playerActionGoodData = actionGoodData[actionGoodData['Player 1'] == playerName]
    
    PassesTable = DataFrame()
    """ Passes / accurate Both Halfs """
    PassesTable = addTableRow(PassesTable,
                            actions1=dataPass[dataPass['Player 1']==playerName],
                            actions1Good=dataPassGood[dataPassGood['Player 1']==playerName],
                            rowName='    Passes / accurate', nSpaces=12)
    
    """ Assist / accurate Both Halfs """
    PassesTable = addTableRow(PassesTable,
                            dataAsist[dataAsist['Player 1']==playerName],
                            '-',
                            '    Assist', 12)
     
    """ 2nd Assist / accurate Both Halfs """
    PassesTable = addTableRow(PassesTable,
                            dataSecondAsist[dataSecondAsist['Player 1']==playerName],
                            '-',
                            '    2nd Assist', 12)
     
    """ key / accurate Both Halfs """
    PassesTable = addTableRow(PassesTable,
                            dataKeyPass[dataKeyPass['Player 1']==playerName],
                            '-',
                            '    Key pass', 12)
    """ progressive PASSES / accurate Both Halfs """
    PassesTable = addTableRow(PassesTable,
                            actions1=dataprogpass[dataprogpass['Player 1']==playerName],
                            actions1Good=dataprogpassGood[dataprogpassGood['Player 1']==playerName],
                            rowName='    progressive Passes / accurate', nSpaces=12)
    
    """ progressive PASSES / into the Final 3rd Both Halfs """
    dP = dataprogpass[dataprogpass['Player 1']==playerName]
    mask = (dP['Actions positions x End'] >= 80) 
    
    dPG = dataprogpassGood[dataprogpassGood['Player 1']==playerName]
    maskG = (dPG['Actions positions x End'] >= 80) 
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    prg pass into the Final 3rd / accurate', nSpaces=12)
    
    """ progressive PASSES / into the box Both Halfs """
    dP = dataprogpass[dataprogpass['Player 1']==playerName]
    mask = (dP['Actions positions x End'] >= 103.5) & \
           (dP['Actions positions y End'] >= 19.9) & \
           (dP['Actions positions y End'] <= 80-19.9)
    
    dPG = dataprogpassGood[dataprogpassGood['Player 1']==playerName]
    maskG = (dPG['Actions positions x End'] >= 103.5) & \
           (dPG['Actions positions y End'] >= 19.9) & \
           (dPG['Actions positions y End'] <= 80-19.9)
    
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    prg pass into the box / accurate', nSpaces=12)
    
    
    """ Throught Pass Both Halfs """
    PassesTable = addTableRow(PassesTable,
                            actions1=dataThroughBall[dataThroughBall['Player 1']==playerName],
                            actions1Good='-',
                            rowName='    Through Pass', nSpaces=12)
    
    """ PASSES / into the box Both Halfs """
    dP = dataPass[dataPass['Player 1']==playerName]
    mask = (dP['Actions positions x End'] >= 103.5) & \
           (dP['Actions positions y End'] >= 19.9) & \
           (dP['Actions positions y End'] <= 80-19.9)
    
    dPG = dataPassGood[dataPassGood['Player 1']==playerName]
    maskG = (dPG['Actions positions x End'] >= 103.5) & \
           (dPG['Actions positions y End'] >= 19.9) & \
           (dPG['Actions positions y End'] <= 80-19.9)
    
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    into the box / accurate', nSpaces=12)
    
    
    """ PASSES / into the Final 3rd Both Halfs """
    dP = dataPass[dataPass['Player 1']==playerName]
    mask = (dP['Actions positions x End'] >= 80) 
    
    dPG = dataPassGood[dataPassGood['Player 1']==playerName]
    maskG = (dPG['Actions positions x End'] >= 80) 
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    into the Final 3rd / accurate', nSpaces=12)
    
    """ Cross / Accurate Both Halfs """
    PassesTable = addTableRow(PassesTable,
                            actions1=dataCross[dataCross['Player 1']==playerName],
                            actions1Good=dataCrossGood[dataCrossGood['Player 1']==playerName],
                            rowName='    Cross / accurate', nSpaces=12)
    
    """ Cross / into the box Both Halfs """
    dP = dataCross[dataCross['Player 1']==playerName]
    mask = (dP['Actions positions x End'] >= 103.5) & \
           (dP['Actions positions y End'] >= 19.9) & \
           (dP['Actions positions y End'] <= 80-19.9)
    
    dPG = dataCross[dataCross['Player 1']==playerName]
    maskG = (dPG['Actions positions x End'] >= 103.5) & \
           (dPG['Actions positions y End'] >= 19.9) & \
           (dPG['Actions positions y End'] <= 80-19.9)
    
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='   Cross into the box / accurate', nSpaces=12)
    
    """ PASSES / Ground Both Halfs """
    dPg = dataPass[dataPass['Event']=='Ground Pass']
    dPgG = dataPassGood[dataPassGood['Event']=='Ground Pass']
    
    PassesTable = addTableRow(PassesTable,
                            actions1=dPg[dPg ['Player 1']==playerName],
                            actions1Good=dPgG[dPgG ['Player 1']==playerName],
                            rowName='    Ground Pass / accurate', nSpaces=12)
    
    """ PASSES / Low Both Halfs """
    dPg = dataPass[dataPass['Event']=='Low Pass']
    dPgG = dataPassGood[dataPassGood['Event']=='Low Pass']
    PassesTable = addTableRow(PassesTable,
                            actions1=dPg[dPg ['Player 1']==playerName],
                            actions1Good=dPgG[dPgG ['Player 1']==playerName],
                            rowName='    Low Pass / accurate', nSpaces=12)
    
    """ PASSES / High Both Halfs """
    dPg = dataPass[dataPass['Event']=='High Pass']
    dPgG = dataPassGood[dataPassGood['Event']=='High Pass']
    PassesTable = addTableRow(PassesTable,
                            actions1=dPg[dPg['Player 1']==playerName],
                            actions1Good=dPgG[dPgG['Player 1']==playerName],
                            rowName='    High Pass / accurate', nSpaces=12)
    
    """ PASSES / Forward Both Halfs """
    dP = dataPass[dataPass['Player 1']==playerName]
    mask = dP['Actions positions x'] < dP['Actions positions x End']
    
    dPG = dataPassGood[dataPassGood['Player 1']==playerName]
    maskG = dPG['Actions positions x'] < dPG['Actions positions x End']
    
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    forward / accurate', nSpaces=12)
        
    """ PASSES / Back Both Halfs """
    dP = dataPass[dataPass['Player 1']==playerName]
    mask = dP['Actions positions x'] >= dP['Actions positions x End']
    
    dPG = dataPassGood[dataPassGood['Player 1']==playerName]
    maskG = dPG['Actions positions x'] >= dPG['Actions positions x End']
        
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    back / accurate', nSpaces=12)
    
    """ PASSES / to the right Both Halfs """
    dP = dataPass[dataPass['Player 1']==playerName]
    mask = dP['Actions positions y'] < dP['Actions positions y End']
        
    dPG = dataPassGood[dataPassGood['Player 1']==playerName]
    maskG = dPG['Actions positions y'] < dPG['Actions positions y End']
    
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    to the right / accurate', nSpaces=12)
    
    """ PASSES / to the left Both Halfs """
    dP = dataPass[dataPass['Player 1']==playerName]
    mask = dP['Actions positions y'] >= dP['Actions positions y End']
    
    dPG = dataPassGood[dataPassGood['Player 1']==playerName]
    maskG = dPG['Actions positions y'] >= dPG['Actions positions y End']
    
    PassesTable = addTableRow(PassesTable,
                            actions1=dP[mask],
                            actions1Good=dPG[maskG],
                            rowName='    to the left / accurate', nSpaces=12)
    
    """ Average Pass length Both Halfs """
    dP = dataPass[dataPass['Player 1']==playerName]
    dPH1 = round(dP[dP['Half']==halfs[0]]['Actions Pos Length'].mean(),1)
    dPH2 = round(dP[dP['Half']==halfs[1]]['Actions Pos Length'].mean(),1)
    print(playerName, dP['Actions Pos Length'].mean(), dPH1, dPH2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    Average Pass length',
                     'Per Match':round(dP['Actions Pos Length'].mean(),1),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    PassesTable = pd.concat([PassesTable, print1], ignore_index=True)
    # دالة تحويل النسب
    def convert_percent_columns(df):
        percent_cols = ["Per Match %", "1st half %", "2nd half %"]
        for col in percent_cols:
            new_vals = []
            for val in df[col]:
                if isinstance(val, str) and "%" in val:
                    try:
                        new_vals.append(float(val.replace("%", "")) / 100)
                    except:
                        new_vals.append(0.0)
                else:
                    new_vals.append(0.0)
            df[col] = new_vals
        return df
    
    
    # معالجة الداتا
    PassesTable = convert_percent_columns(PassesTable)
   
    # ========================= Display Table ==========================
    #st.dataframe(PassesTable)   # لو داخل Streamlit
    # ==================== Plot Table ====================
    from PIL import Image
    # رسم الجدول
    fig_PassTable, ax_PassTable = plt.subplots(figsize=(16, 16))
    
    img = Image.open(r"WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg")  # حط هنا مسار الصورة
    fig_PassTable.figimage(img, xo=600, yo=450, alpha=0.2, zorder=0)
    row_colors = {
        "top4": "#1b1f1f",       # غامق جدًا
        "top6": "#2e3a3a",       # غامق
        "playoffs": "#555d55",   # متوسط غامق
        "relegation": "#6b5e4d", # غامق بني/رمادي
        "even": "#3d4949",       # غامق للصفوف الزوجية
        "odd": "#4a5656",        # غامق للصفوف الفردية
    }
    bg_color =  "w"  # row_colors["odd"]
    text_color = "k"
    
    plt.rcParams["text.color"] = text_color
    plt.rcParams["font.family"] = "Arial"
    
    fig_PassTable.set_facecolor(bg_color)
    ax_PassTable.set_facecolor(bg_color)
    
    
    tab = Table(
        PassesTable,
        cell_kw={"linewidth": 0, "edgecolor": "k","height":1},
        textprops={"ha": "right","va":"center","fontsize":12},
        col_label_divider=True,  # إزالة الخط الفاصل
        col_label_divider_kw={"color": "gray", "lw": .45},
        index_col="PATASTATS INDEX",
        even_row_color="w",
        footer_divider=False,
        footer_divider_kw={"color": bg_color, "lw": .5},
        row_divider_kw={"color": "lightgray", "lw": .5},
        column_border_kw={"color": "darkred", "lw": .5},
    
        column_definitions=[
            ColumnDefinition("PATASTATS INDEX",title="", textprops={"ha": "left", "fontsize": 14},width=1.95),
    
            # النصوص الأصلية
            ColumnDefinition("Per Match",title="Per 90",textprops={"ha": "center", "fontsize": 12}),
            ColumnDefinition("1st half"),
            ColumnDefinition("2nd half"),
    
            # أعمدة الدونات مع تكبير الحجم
            ColumnDefinition(
                "Per Match %",
                title="Per 90 %",
                width=.7,
                textprops={"ha": "center"},
                plot_fn=progress_donut, 
                plot_kw={ "is_pct": True, "formatter": "{:.0%}", "radius": 0.5,"color":"r" , "width": 0.05 ,"alpha":.80}
            ),
            ColumnDefinition(
                "1st half %",
                width=.7,
                textprops={"ha": "center"},
                plot_fn=progress_donut, 
                plot_kw={"is_pct": True, "formatter": "{:.0%}", "radius": 0.5, "color":"r" ,"width": 0.05,"alpha":.80}
            ),
            ColumnDefinition(
                "2nd half %", 
                width=.7, 
                textprops={"ha": "center"},
                plot_fn=progress_donut, 
                plot_kw={ "is_pct": True, "formatter": "{:.0%}", "radius": 0.5,"color":"r" ,"width": 0.05,"alpha":.80 }
            ),
        ],
    )
    
    fig_PassTable.text(
        0.14, 0.9,                  # إحداثيات x و y
        "\nPassing Stats",
        fontsize=22,
        color='k'                    # لون الجزء الأول
    )
    fig_PassTable.text(
        0.14, 0.892,                  # إحداثيات x و y
        "____________",        # الجزء الأول
        fontsize=22,
        color='gold'                    # لون الجزء الأول
    )
    
    
    plt.show()
    #st.pyplot(fig_PassTable)
    
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as path_effects
    
    # إنشاء colormap شبيه بـ StatsBomb بـ 20 لون من الأزرق للأبيض للأحمر
    statsbomb_cmap_red_blue =   LinearSegmentedColormap.from_list(
        "Blue-Gray-Red",
        ['#000B2B',  # أزرق غامق أصلي
         '#929591',  # رمادي متوسط
         '#8C000F'], # أحمر غامق
        N=5  # لتدرج سلس
    )    

    from mplsoccer import VerticalPitch
    import matplotlib.pyplot as plt
    from io import BytesIO
    from matplotlib.ticker import StrMethodFormatter
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    
    # فلترة البيانات
    actionsPlayerGood  = actionGoodData[actionGoodData['Player 1']==playerName]
    actionsPlayerBad   = actionBadData[actionBadData['Player 1']==playerName]
    
    # إنشاء الشكل
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # إعداد الملعب الرأسي
    pitch = Pitch(pitch_color='w', line_color='k',line_zorder=2)
    pitch.draw(ax=ax)
    
    # رسم الأفعال الجيدة
    pitch.scatter(
        actionsPlayerGood['Actions positions x'],
        actionsPlayerGood['Actions positions y'],
        c='lime', s=80, ax=ax, label='Good Actions',zorder=2
    )
    
    # رسم الأفعال السيئة
    pitch.scatter(
        actionsPlayerBad['Actions positions x'],
        actionsPlayerBad['Actions positions y'],
        c='red', s=80, marker='x', ax=ax, label='Bad Actions',zorder=2
    )
    # heatmap and labels
    
    path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                path_effects.Normal()]
    
    bin_statistic = pitch.bin_statistic(playerData['Actions positions x'], playerData['Actions positions y'], statistic='count',
                                                    normalize=True,bins=(6,4))
    hm=pitch.heatmap_positional([bin_statistic], ax=ax,
                             cmap=statsbomb_cmap_red_blue, edgecolors='None',zorder=1,alpha=.7)
    labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=12,
                                 ax=ax, ha='center', va='center',
                                 str_format='{:.0%}', path_effects=path_eff)
    
    
    # إضافة colorbar
    cax = fig.add_axes([0.62, 0.05, 0.15, 0.01])
    cbar = fig.colorbar(hm[0], cax=cax,ax=ax, fraction=0.016, pad=0.03,orientation='horizontal',
        format=StrMethodFormatter("{x:.0%}"   ))# هنا بيعرض القيم كنسبة مئوية بدون كسور عشرية
    cbar.set_label('Action Intensity', fontsize=14)
      # [left, bottom, width, height]
    
    #ax_Home_logo = add_image(img, fig, left=0.04,
    #                         bottom=axs['title'].get_position().y0,
    #                         height=axs['title'].get_position().height * 1.5)
    pitch.inset_image(60.5, 45, img, height=70, alpha=.2, ax=ax, zorder=-1)
    pitch.inset_image(115, -15, img, height=25, alpha=1, ax=ax, zorder=1)
    
    
    # عنوان
    ax.set_title(f"\n\nHeatmap of All Player Actions", fontsize=35, color='gold', y=1.0,x=.4)
    
    
    arrow = FancyArrowPatch(
        (40, 82),  # البداية
        (80, 82),  # النهاية
        arrowstyle='->',
        linewidth=1,
        color='k',
        mutation_scale=10,zorder=2
    )
    ax.add_patch(arrow)
    
    # إضافة نص بجانب السهم يشير إلى منطقة الهجوم
    ax.annotate(
        'Attack Direction',
        xy=(60, 85),  # منتصف السهم أفقيًا وتحت الملعب عموديًا
        ha='center',
        color='k',
        fontsize=14,
        zorder=2
    )
    # وسوم
    ax.legend(loc='upper right', fontsize=12, facecolor='black', edgecolor='white', labelcolor='white')
    
    
    legend = ax.legend( title='', labelspacing=2, loc="upper center", ncol=2, 
          frameon=False, fancybox=True, shadow=True, bbox_to_anchor=(0.2, -0.06), markerscale=1.5 ,title_fontsize=34)
    for text in legend.get_texts():
        text.set_fontsize(14)
        text.set_c('k')
    
    
    # ضبط حدود الملعب
    ax.set_xlim(-10, 130)
    ax.set_ylim(85, -15) 
    # حفظ الصورة
    playerActionsImage = BytesIO()
    fig.savefig(playerActionsImage, format='png', edgecolor='white', bbox_inches='tight', pad_inches=0, transparent=False)
    #st.pyplot(fig)
    playerActionsImage.seek(0)
    plt.close('all')
    

    from mplsoccer import VerticalPitch
    import matplotlib.pyplot as plt
    from io import BytesIO
    #from IPython.display import Image, display
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import StrMethodFormatter
    from highlight_text import ax_text
    
    # ================================
    # 1. فلترة بيانات اللاعب
    # ================================
    playerPassGood = dataPassGood[dataPassGood['Player 1'] == playerName]
    playerPassBad  = dataPassBad[dataPassBad['Player 1'] == playerName]
    playerDataKeyPass  = dataKeyPass[dataKeyPass['Player 1'] == playerName]
    playerAssist  = dataAsist[dataAsist['Player 1'] == playerName]
    player2ndAssist  = dataSecondAsist[dataSecondAsist['Player 1'] == playerName]
    
    halves = ['1st Half', '2nd Half']
    
    def filter_half(data, half):
        return data[data['Half'] == half]
    
    # ================================
    # 2. إنشاء الشكل والمحاور
    # ================================
    fig_passes, axs_passes = plt.subplots(1, 2, figsize=(16, 9))
    plt.subplots_adjust(
        left=0.1,
        right=.99,
        top=0.75,
        bottom=0.1,
        wspace=0.005
    )
    pitch = VerticalPitch(pitch_color='w', line_color='k', line_zorder=2)
    
    # ================================
    # 3. تقسيم البيانات حسب الشوط
    # ================================
    def prepare_half_data(half):
        pass_good = filter_half(playerPassGood, half)
        pass_bad = filter_half(playerPassBad, half)
        key_pass = filter_half(playerDataKeyPass, half)
        assist = filter_half(playerAssist, half)
        second_assist = filter_half(player2ndAssist, half)
        
        pass_good = pass_good.drop(index=key_pass.index, errors='ignore')
        pass_bad = pass_bad.drop(index=key_pass.index, errors='ignore')
        
        return pass_good, pass_bad, key_pass, assist, second_assist
    
    half_data = [prepare_half_data(h) for h in halves]
    
    # ================================
    # 4. رسم خطوط التمريرات
    # ================================
    def draw_pass_lines(ax, pass_good, pass_bad, key_pass, assist, second_assist):
        pitch.draw(ax=ax)
        pitch.lines(pass_good['Actions positions x'], pass_good['Actions positions y'],
                    pass_good['Actions positions x End'], pass_good['Actions positions y End'],
                    lw=3, transparent=True, comet=True, color='#56ae6c', zorder=3, ax=ax, label='Completed passes')
        pitch.lines(pass_bad['Actions positions x'], pass_bad['Actions positions y'],
                    pass_bad['Actions positions x End'], pass_bad['Actions positions y End'],
                    lw=3, transparent=True, comet=True, color='#ba4f45', zorder=3, ax=ax, label='Incomplete passes')
        pitch.lines(key_pass['Actions positions x'], key_pass['Actions positions y'],
                    key_pass['Actions positions x End'], key_pass['Actions positions y End'],
                    lw=3, transparent=True, comet=True, color='#7E1E9C', zorder=3, ax=ax, label='Key passes')
        pitch.lines(assist['Actions positions x'], assist['Actions positions y'],
                    assist['Actions positions x End'], assist['Actions positions y End'],
                    lw=3, transparent=True, comet=True, color='gold', zorder=3, ax=ax, label='Assist')
        pitch.lines(second_assist['Actions positions x'], second_assist['Actions positions y'],
                    second_assist['Actions positions x End'], second_assist['Actions positions y End'],
                    lw=3, transparent=True, comet=True, color='orange', zorder=3, ax=ax, label='2nd Assist')
    
    for i, data in enumerate(half_data):
        draw_pass_lines(axs_passes[i], *data)
        axs_passes[i].set_title(f"{halves[i]}", y=1.05, fontsize=22, color='k')
    
    # ================================
    # 5. دالة Heatmap جديدة (أفقي وعمودي)
    # ================================
    def draw_heatmap_full(ax, data, pitch, cmap):
        # Heatmap أفقي
        bin_stat_h = pitch.bin_statistic_positional(
            data['Actions positions x End'],
            data['Actions positions y End'],
            statistic='count',
            positional='horizontal',
            normalize=True
        )
        hm_h = pitch.heatmap_positional(bin_stat_h, ax=ax, cmap=cmap, edgecolors='None', zorder=2, alpha=.25)
        
        bin_dict_h = bin_stat_h[0]
        heat_values_h = bin_dict_h['statistic'].flatten()
        x_centers_h = bin_dict_h['cx'].flatten()
        y_centers_h = bin_dict_h['cy'].flatten()
        norm_h = mcolors.Normalize(vmin=heat_values_h.min(), vmax=heat_values_h.max())
        
        for x, y, val in zip(x_centers_h, y_centers_h, heat_values_h):
            color = cmap(norm_h(val))
            ax.text(y, 126.5, f'{val:.0%}', ha='center', va='center', fontsize=12, color=color)
            ax.text(y, -6.5, f'{val:.0%}', ha='center', va='center', fontsize=12, color=color)
        
        y_edges = bin_dict_h['y_grid']
        y_top = y_edges[1:-1, 1]
        for y in y_top:
            ax.text(y, 125, "|", ha='center', va='bottom', fontsize=15, color='gray')
            ax.text(y, -5, "|", ha='center', va='bottom', fontsize=15, color='gray')
        
        # Heatmap عمودي
        bin_stat_v = pitch.bin_statistic_positional(
            data['Actions positions x End'],
            data['Actions positions y End'],
            statistic='count',
            positional='vertical',
            normalize=True
        )
        
        bin_dict_v = bin_stat_v[0]
        heat_values_v = bin_dict_v['statistic'].flatten()
        x_centers_v = bin_dict_v['cx'].flatten()
        y_centers_v = bin_dict_v['cy'].flatten()
        norm_v = mcolors.Normalize(vmin=heat_values_v.min(), vmax=heat_values_v.max())
        
        for x, y, val in zip(x_centers_v, y_centers_v, heat_values_v):
            color = cmap(norm_v(val))
            ax.text(-6.5, x, f'{val:.0%}', ha='center', va='center', fontsize=12, color=color)
            ax.text(86.5, x, f'{val:.0%}', ha='center', va='center', fontsize=12, color=color)
        
        x_edges = bin_dict_v['x_grid']
        x_top = x_edges[0, 1:-1]
        for x in x_top:
            ax.text(-5.5, x, "|", ha='center', va='bottom', fontsize=15, color='gray', rotation=90)
            ax.text(85.5, x, "|", ha='center', va='bottom', fontsize=15, color='gray', rotation=90)
        
        return hm_h
    
    heatmaps = [draw_heatmap_full(axs_passes[i], data[0], pitch, statsbomb_cmap_red_blue) for i, data in enumerate(half_data)]
    
    # ================================
    # 6. صورة خلفية (اختياري)
    # ================================
    ax_bg = fig_passes.add_axes([0.8, .82, .15, .18])
    ax_bg.imshow(img, aspect='auto', alpha=1)
    ax_bg.axis('off')
    
    # ================================
    # 7. نص أسفل الشكل
    # ================================
    complete_pass = len(playerPassGood)
    total_pass = len(playerPassGood) + len(playerPassBad)
    if total_pass == 0:
        pass_percentage = "0%"
    else:
        pass_percentage = f'{round(100 * complete_pass / total_pass, 1)}%'
    KEY_pass = len(playerDataKeyPass)
    Assist_pass = len(playerAssist)
    Assist2nd_pass = len(player2ndAssist)
    
    SUB_TEXT = (f'<Types of passes:>  '
                f'<Accurate pass: {complete_pass}>/{total_pass} | '
                f'<{pass_percentage} |>'
                f'< Assist : {Assist_pass} |>'
                f'< 2nd Assist: {Assist2nd_pass} |> '
                f'< key Pass: {KEY_pass}> \n\n')
    
    ax_h = fig_passes.add_axes([0, 0, 1, 1])
    ax_h.axis('off')
    
    highlight_text_props = [
        {'color': 'k'},
        {'color': '#56ae6c'},
        {'color': '#56ae6c'},
        {'color':'gold'},
        {'color':'orange'},
        {'color':"#7E1E9C"}
    ]
    
    ax_text(0.5, 0.02, SUB_TEXT, ax=ax_h, highlight_textprops=highlight_text_props,
            ha='center', va='center', fontsize=22)
    
    # ================================
    # 8. Colorbar
    # ================================
    cax = fig_passes.add_axes([0.1, 0.15, 0.04, 0.55])
    cbar = fig_passes.colorbar(heatmaps[0][0], cax=cax, ax=ax, location="left", fraction=0.016, pad=0.0,
                               orientation='vertical', format=StrMethodFormatter("{x:.0%}"))
    cbar.set_label('Accurate Pass Intensity', fontsize=20, labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    
    # ================================
    # 9. عنوان الشكل العام
    # ================================
    fig_passes.suptitle("Pass Map and Pass Types", fontsize=40, color='gold', y=.95, x=0.3)
    
    # ================================
    # 10. سهم اتجاه الهجوم
    # ================================
    arrow = FancyArrowPatch((0.55, 0.16), (0.55, 0.26), arrowstyle='-', linewidth=2,
                            color='k', mutation_scale=20, transform=fig_passes.transFigure, zorder=2)
    fig_passes.patches.append(arrow)
    
    arrow1 = FancyArrowPatch((0.55, 0.56), (0.55, 0.66), arrowstyle='->', linewidth=2,
                             color='k', mutation_scale=20, transform=fig_passes.transFigure, zorder=2)
    fig_passes.patches.append(arrow1)
    
    fig_passes.text(0.55, 0.31, "Attack Direction", ha='center', va='bottom',
                    fontsize=18, color='k', zorder=3, rotation=90)
    
    # ================================
    # 11. عرض الشكل
    # ================================
    plt.show()
    #st.pyplot(fig_passes)

    
    shootingTable = DataFrame()
    
    shotInBoxMask = (dataShot['Actions positions x']>=103.5)&(dataShot['Actions positions y']>=19.9)&(dataShot['Actions positions y']<=80-19.9)
    
    shotInBoxMaskGood = (dataShotGood['Actions positions x']>=103.5)&(dataShotGood['Actions positions y']>=19.9)&(dataShotGood['Actions positions y']<=80-19.9)
    
    shotsfromFreekick = dataShot[dataShot['Extra 1']=="Free Kick"]
    shotsfromFreekickGood = dataShotGood[dataShotGood['Extra 1']=="Free Kick"]
    
    PenaltyShot = dataShot[dataShot['Extra 1']=="Penalty"]
    PenaltyShotGood = PenaltyShot[PenaltyShot['Outcome']=='Goal']
    
    shotOutOFBoxMask = dataShot[~shotInBoxMask]
    shotOutOFBoxMaskGood = dataShotGood[~shotInBoxMaskGood]
    # إدخال الصف في الجدول
    
    
    
    
    shootingTable = addTableRow(shootingTable, 
        dataShot[(dataShot['Player 1']==playerName) & (dataShot['Outcome']=="Goal")], 
        '-', 
        '    Goals', 12)
    
    shootingTable = addTableRow(shootingTable, dataShot[dataShot['Player 1']==playerName],
                              dataShotGood[dataShotGood['Player 1']==playerName], '    Shots / on Target', 12)
    
    # Shots inside the box
    shootingTable = addTableRow(
        shootingTable,
        dataShot[shotInBoxMask & (dataShot['Player 1']==playerName)],
        dataShotGood[shotInBoxMaskGood & (dataShotGood['Player 1']==playerName)],
        '    Shots in Box / on Target',
        12
    )
    
    # Shots outside the box
    shootingTable = addTableRow(
        shootingTable,
        dataShot[~shotInBoxMask & (dataShot['Player 1']==playerName)],
        dataShotGood[~shotInBoxMaskGood & (dataShotGood['Player 1']==playerName)],
        '    Shots Out OF Box / on Target',
        12
    )
    
    
    shootingTable = addTableRow(
        shootingTable,
        shotsfromFreekick[shotsfromFreekick['Player 1']==playerName],
        shotsfromFreekickGood[shotsfromFreekickGood['Player 1']==playerName],
        '    Free Kicks Shots / on Target',
        12
    )
    
    shootingTable = addTableRow(
        shootingTable,
        PenaltyShot[PenaltyShot['Player 1']==playerName],
        PenaltyShotGood[PenaltyShotGood['Player 1']==playerName],
        '    Penalty / Goal',
        12
    )
    
    
    
    
    shootingTable = addTableRow(shootingTable, dataShotGood[(dataShotGood['Player 1']==playerName)],
                              dataShotGood[(dataShotGood['Player 1']==playerName)&(dataShotGood['Extra 2']=='Head')], '    Shots on Target / Head', 12)
    
    shootingTable = addTableRow(shootingTable, dataShotGood[(dataShotGood['Player 1']==playerName)],
                              dataShotGood[(dataShotGood['Player 1']==playerName)&(dataShotGood['Extra 2']=='Right Foot')], '    Shots on Target / R Foot', 12)
    
    shootingTable = addTableRow(shootingTable, dataShotGood[(dataShotGood['Player 1']==playerName)],
                              dataShotGood[(dataShotGood['Player 1']==playerName)&(dataShotGood['Extra 2']=='Left Foot')], '    Shots on Target / L Foot', 12)
    
    
    """ Goal conversion ratio Both Halfs """
    # إجمالي نسبة الأهداف من التسديدات (للمباراة كلها)
    dP_total = len(dataShot[(dataShot['Player 1'] == playerName) & (dataShot['Outcome'] == "Goal")]) / \
               max(1,len(dataShot[dataShot['Player 1'] == playerName]))
    
    # النسبة في الشوط الأول
    dP_H1 = len(dataShot[(dataShot['Player 1'] == playerName) & 
                         (dataShot['Outcome'] == "Goal") & 
                         (dataShot['Half'] == halfs[0])]) / \
            max(1,len(dataShot[(dataShot['Player 1'] == playerName) & 
                         (dataShot['Half'] == halfs[0])]))
    
    # النسبة في الشوط الثاني
    dP_H2 = len(dataShot[(dataShot['Player 1'] == playerName) & 
                         (dataShot['Outcome'] == "Goal") & 
                         (dataShot['Half'] == halfs[1])]) / \
            max(1,len(dataShot[(dataShot['Player 1'] == playerName) & 
                         (dataShot['Half'] == halfs[1])]))
    
    print(playerName, round(dP_total, 2), round(dP_H1, 2), round(dP_H2, 2))
    
    print_ratio = pd.DataFrame({
        'PATASTATS INDEX': '    Goal Conversion Ratio',
        'Per Match': round(dP_total, 2),
        '1st half': round(dP_H1, 2),
        '2nd half': round(dP_H2, 2),
    }, index=[0])
    
    shootingTable = pd.concat([shootingTable, print_ratio], ignore_index=True)
    
    """ Total XG Shot  Both Halfs """
    dP = dataShot[dataShot['Player 1']==playerName]
    dPH1 = round(dP[dP['Half']==halfs[0]]['xG'].sum(),2)
    dPH2 = round(dP[dP['Half']==halfs[1]]['xG'].sum(),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    XG ',
                     'Per Match':round(dP['xG'].sum(),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    """  XGoT Shot  Both Halfs """
    dP = dataShotGood[dataShotGood['Player 1']==playerName]
    dPH1 = round(dP[dP['Half']==halfs[0]]['xG'].sum(),2)
    dPH2 = round(dP[dP['Half']==halfs[1]]['xG'].sum(),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    XGoT ',
                     'Per Match':round(dP['xG'].sum(),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    """ np XG  Both Halfs """
    dP = dataShot[(dataShot['Player 1'] == playerName) & (dataShot['Extra 1']!="Penalty")]
    dPH1 = round(dP[dP['Half']==halfs[0]]['xG'].sum(),2) 
    dPH2 = round(dP[dP['Half']==halfs[1]]['xG'].sum(),2) 
        
    print1 = DataFrame({'PATASTATS INDEX':'    np XG  ',
                     'Per Match':round(dP['xG'].sum(),2) ,
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    
    
    """  XGoT Head Shot  Both Halfs """
    dP = dataShotGood[(dataShotGood['Player 1']==playerName)&(dataShotGood['Extra 2']=='Head')]
    dPH1 = round(dP[dP['Half']==halfs[0]]['xG'].sum(),2)
    dPH2 = round(dP[dP['Half']==halfs[1]]['xG'].sum(),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    XGoT By Head ',
                     'Per Match':round(dP['xG'].sum(),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    """  XGoT Right Foot Shot  Both Halfs """
    dP = dataShotGood[(dataShotGood['Player 1']==playerName)&(dataShotGood['Extra 2']=='Right Foot')]
    dPH1 = round(dP[dP['Half']==halfs[0]]['xG'].sum(),2)
    dPH2 = round(dP[dP['Half']==halfs[1]]['xG'].sum(),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    XGoT By R Foot ',
                     'Per Match':round(dP['xG'].sum(),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    
    """  XGoT Right Foot Shot  Both Halfs """
    dP = dataShotGood[(dataShotGood['Player 1']==playerName)&(dataShotGood['Extra 2']=='Left Foot')]
    dPH1 = round(dP[dP['Half']==halfs[0]]['xG'].sum(),2)
    dPH2 = round(dP[dP['Half']==halfs[1]]['xG'].sum(),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    XGoT By L Foot ',
                     'Per Match':round(dP['xG'].sum(),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    
    
    """ np XG  / Shot  Both Halfs """
    
    dP = dataShot[(dataShot['Player 1'] == playerName) & (dataShot['Extra 1']!="Penalty")]
    dPH1 = round(dP[dP['Half']==halfs[0]]['xG'].sum() / max(1,len(dP[dP['Half']==halfs[0]])),2)
    dPH2 = round(dP[dP['Half']==halfs[1]]['xG'].sum() / max(1,len(dP[dP['Half']==halfs[1]])),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    np XG per shot ',
                     'Per Match':round(dP['xG'].sum() / max(1, len(dP)),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    
    """ Gaol - XG  Both Halfs """
    
    dP = dataShot[(dataShot['Player 1'] == playerName) & (dataShot['Outcome']=="Goal")] 
    dp2 = dataShot[dataShot['Player 1'] == playerName]
    dPH1 = len(dP[dP['Half']==halfs[0]]) - round(dp2[dp2['Half']==halfs[0]]['xG'].sum(),2)
    dPH2 = len(dP[dP['Half']==halfs[1]]) - round(dp2[dp2['Half']==halfs[1]]['xG'].sum(),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    Goals - XG ',
                     'Per Match':len(dP) - round(dp2['xG'].sum(),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable.round(2), print1], ignore_index=True)
    
    
    """ np Gaols - np XG  Both Halfs """
    
    dP = dataShot[(dataShot['Player 1'] == playerName) & (dataShot['Outcome']=="Goal") & (dataShot['Extra 1']!="Penalty")] 
    dp2 = dataShot[(dataShot['Player 1'] == playerName) & (dataShot['Extra 1']!="Penalty")]
    dPH1 = len(dP[dP['Half']==halfs[0]]) - round(dp2[dp2['Half']==halfs[0]]['xG'].sum(),2)
    dPH2 = len(dP[dP['Half']==halfs[1]]) - round(dp2[dp2['Half']==halfs[1]]['xG'].sum(),2)
        
    print1 = DataFrame({'PATASTATS INDEX':'    np Goals - np XG ',
                     'Per Match':len(dP) - round(dp2['xG'].sum(),2),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable.round(2), print1], ignore_index=True)
    
    
    """ Average Shot length Both Halfs """
    dP = dataShot[dataShot['Player 1']==playerName]
    dPH1 = round(dP[dP['Half']==halfs[0]]['Actions Pos Length'].mean(),1)
    dPH2 = round(dP[dP['Half']==halfs[1]]['Actions Pos Length'].mean(),1)
        
    print1 = DataFrame({'PATASTATS INDEX':'    Average Shot Distance ',
                     'Per Match':round(dP['Actions Pos Length'].mean(),1),
                     '1st half':dPH1,
                     '2nd half':dPH2,
                     }, index=[0])
    shootingTable = pd.concat([shootingTable, print1], ignore_index=True)
    
    shootingTable = convert_percent_columns(shootingTable)
    shootingTable
 
    from PIL import Image
    # رسم الجدول
    fig_ShotTable, ax_ShotTable = plt.subplots(figsize=(16, 16))
    
    img = Image.open(r"WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg")  # حط هنا مسار الصورة
    fig_ShotTable.figimage(img, xo=600, yo=450, alpha=0.2, zorder=0)
    row_colors = {
        "top4": "#1b1f1f",       # غامق جدًا
        "top6": "#2e3a3a",       # غامق
        "playoffs": "#555d55",   # متوسط غامق
        "relegation": "#6b5e4d", # غامق بني/رمادي
        "even": "#3d4949",       # غامق للصفوف الزوجية
        "odd": "#4a5656",        # غامق للصفوف الفردية
    }
    bg_color =  "w"  # row_colors["odd"]
    text_color = "k"
    
    plt.rcParams["text.color"] = text_color
    plt.rcParams["font.family"] = "Arial"
    
    fig_ShotTable.set_facecolor(bg_color)
    ax_ShotTable.set_facecolor(bg_color)
    
    def safe_float(x):
        try:
            val = float(x)
            if val < 0:
                return f"- {abs(val):.2f}"  # مسافة بعد السالب
            else:
                return f"{val:.2f}"
        except:
            return str(x)  # يسيب النص كما هو
    
    num_cols = ["Per Match", "1st half", "2nd half"]
    for col in num_cols:
        shootingTable[col] = shootingTable[col].apply(safe_float)
    
    tab = Table(
        shootingTable.round(2),
        cell_kw={"linewidth": 0, "edgecolor": "k","height":1.2},
        textprops={"ha": "right","va":"center","fontsize":12},
        col_label_divider=True,  # إزالة الخط الفاصل
        col_label_divider_kw={"color": "gray", "lw": .45},
        index_col="PATASTATS INDEX",
        even_row_color="w",
        footer_divider=False,
        footer_divider_kw={"color": bg_color, "lw": .5},
        row_divider_kw={"color": "lightgray", "lw": .5},
        column_border_kw={"color": "darkred", "lw": .5},
    
        column_definitions=[
            ColumnDefinition("PATASTATS INDEX",title="", textprops={"ha": "left", "fontsize": 14},width=1.95),
    
            # النصوص الأصلية
            ColumnDefinition("Per Match",title="Per 90",textprops={"ha": "center", "fontsize": 12}),
            ColumnDefinition("1st half"),
            ColumnDefinition("2nd half"),
    
            # أعمدة الدونات مع تكبير الحجم
            ColumnDefinition(
                "Per Match %",
                title="Per 90 %",
                width=.7,
                textprops={"ha": "center"},
                plot_fn=progress_donut, 
                plot_kw={ "is_pct": True, "formatter": "{:.0%}", "radius": 0.49,"color":"r" , "width": 0.05 ,"alpha":.80}
            ),
            ColumnDefinition(
                "1st half %",
                width=.7,
                textprops={"ha": "center"},
                plot_fn=progress_donut, 
                plot_kw={"is_pct": True, "formatter": "{:.0%}", "radius": 0.49, "color":"r" ,"width": 0.05,"alpha":.80}
            ),
            ColumnDefinition(
                "2nd half %", 
                width=.7, 
                textprops={"ha": "center"},
                plot_fn=progress_donut, 
                plot_kw={ "is_pct": True, "formatter": "{:.0%}", "radius": 0.49,"color":"r" ,"width": 0.05,"alpha":.80 }
            ),
        ],
    )
    
    fig_ShotTable.text(
        0.14, 0.9,                  # إحداثيات x و y
        "\nShooting Stats",
        fontsize=22,
        color='k'                    # لون الجزء الأول
    )
    fig_ShotTable.text(
        0.14, 0.892,                  # إحداثيات x و y
        "____________",        # الجزء الأول
        fontsize=22,
        color='gold'                    # لون الجزء الأول
    )
    
    
    plt.show()

    #st.pyplot(fig_ShotTable)

    from mplsoccer import VerticalPitch
    import matplotlib.pyplot as plt
    from io import BytesIO
    #from IPython.display import Image, display
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import StrMethodFormatter
    from highlight_text import ax_text
    
    # ================================
    # 1. فلترة بيانات اللاعب
    # ================================
    playerShots = dataShot[dataShot['Player 1'] == playerName]
    
    # ================================
    # تعريف الألوان لكل نوع قدم
    # ================================
    colors = {'Blocked':'gray','Off T':'r','Saved':'gold','Wayward':'none','Post':'gold'}
    edgecolors = {'Blocked':'k','Off T':'none','Saved':'gold','Wayward':'r','Post':'gold'}
    
    # تعريف الـ marker لكل Outcome
    markers = {'Goal':'football','Blocked':'s','Off T':'o','Saved':'o','Wayward':'o','Post':'d'}
    
    # تعريف linestyle لكل Outcome
    linestyles = {'Goal':'-','Blocked':'-','Off T':'--','Saved':'-','Wayward':'--','Post':'-'}
    
    # تعريف labels لكل Outcome
    labels = {'Goal':'Goal','Blocked':'Blocked','Off T':'Off T','Saved':'Saved','Wayward':'Wayward','Post':'Post'}
    
    # ================================
    # 2. إنشاء الشكل والمحاور
    # ================================
    fig_Shots, axs_Shots = plt.subplots(figsize=(16, 14.25))
    
    pitch = VerticalPitch(
        half=True,
        pitch_color='w',
        goal_type='box',
        positional=True,
        positional_color='k',
        positional_alpha=.2,
        line_color='k', 
        line_zorder=2
    )
    
    # ================================
    # 4. رسم خطوط التمريرات
    # ================================
    def draw_Shots_scatter(ax, playerShots):
        pitch.draw(ax=ax)
        added_labels = set()  # لتجنب تكرار الـ legend
        playerShots = playerShots.dropna(subset=['Actions positions x', 'Actions positions y'])

        for outcome, group in playerShots.groupby('Outcome'):
            for i, row in group.iterrows():
                xg_value = row.get('xG', 0)
                mark = markers.get(outcome, 'o')
                label_text = labels.get(outcome, '-') if outcome not in added_labels else None
                linestyle_ = linestyles.get(outcome, '-')  # الخطوط العادية فقط للماركرز العادية
                color = colors.get(outcome, 'none')
                edgecolor = edgecolors.get(outcome, 'none')
            
                if mark == 'football':
                    
                    try:
                        
                        pitch.scatter(
                        x=row['Actions positions x'],
                        y=row['Actions positions y'],
                        s=xg_value*800 if pd.notna(xg_value) else 400,
                        hexcolor='w',
                        pentcolor='k',
                        label=label_text,
                        lw=1,
                        zorder=4,
                        ax=ax
                        )
                    except Exception as e:
                        print(f"⚠️ Football marker error for row {i}: {e}")
                        # fallback marker to avoid crash
                        pitch.scatter(
                        x=row['Actions positions x'],
                        y=row['Actions positions y'],
                        s=xg_value*800 if pd.notna(xg_value) else 400,
                        color='g',
                        edgecolor='w',
                        marker='d',
                        label=label_text,
                        lw=1,
                        zorder=4,
                        ax=ax
                        )

                else:
                    # ماركرات عادية: يمكن تمرير color, edgecolor, linestyle
                    pitch.scatter(
                        x=row['Actions positions x'],
                        y=row['Actions positions y'],
                        s=xg_value*800,
                        marker=mark,
                        color=color,
                        edgecolor=edgecolor,
                       # linestyle=linestyle_,
                        lw=1,
                        label=label_text,
                        zorder=4,
                        ax=ax
                    )

                # رسم الأسهم إذا موجودة
                if pd.notna(row.get('Actions positions x End')) and pd.notna(row.get('Actions positions y End')):
                    pitch.arrows(
                        xstart=row['Actions positions x'] + 0.5,
                        ystart=row['Actions positions y'],
                        xend=row['Actions positions x End'],
                        yend=row['Actions positions y End'],
                        width=xg_value*20,
                        color='lime',
                        alpha=min(xg_value*2, 1),  # Alpha بين 0 و 1
                        ax=ax
                    )
            added_labels.add(outcome)

    draw_Shots_scatter(axs_Shots, playerShots)
    
    # ================================
    # 5. دالة Heatmap جديدة (أفقي وعمودي)
    # ================================
    def draw_heatmap_full(ax, data, pitch, cmap):
        bin_stat_h = pitch.bin_statistic_positional(
            data['Actions positions x'],
            data['Actions positions y'],
            statistic='count',
            positional='horizontal',
            normalize=True
        )
        
        bin_dict_h = bin_stat_h[0]
        heat_values_h = bin_dict_h['statistic'].flatten()
        x_centers_h = bin_dict_h['cx'].flatten()
        y_centers_h = bin_dict_h['cy'].flatten()
        norm_h = mcolors.Normalize(vmin=heat_values_h.min(), vmax=heat_values_h.max())
        
        for x, y, val in zip(x_centers_h, y_centers_h, heat_values_h):
            color = cmap(norm_h(val))
            ax.text(y, 125.5, f'{val:.0%}', ha='center', va='center', fontsize=12, color=color, zorder=2)
        
        y_edges = bin_dict_h['y_grid']
        y_top = y_edges[1:-1, 1]
        for y in y_top:
            ax.text(y, 124, "|", ha='center', va='bottom', fontsize=15, color='gray', zorder=2)
    
    heatmaps = [draw_heatmap_full(axs_Shots, playerShots, pitch, statsbomb_cmap_red_blue)]
    
    # ================================
    # 6. خطوط ومتوسط المسافة
    # ================================
    if not playerShots['Actions positions x'].dropna().empty:
        avg_ds_shot = np.nanmean(playerShots['Actions positions x'])  # يتجاهل NaN
        avg_ds_shot = round(avg_ds_shot)
    else:
        avg_ds_shot = 0  # أو أي قيمة افتراضية لو مفيش تسديدات    
    pitch.lines(
        120, 75,
        avg_ds_shot, 75,
        lw=3,
        linestyle='--',
        transparent=True,
        comet=True,
        label='Avg Shot Distance',
        color='gold',
        ax=axs_Shots
    )
    
    axs_Shots.annotate(
        f'Avg Shot Distance {120-avg_ds_shot}Y',
        xy=(78, avg_ds_shot+2),
        zorder=2,
        ha='center',
        color='#7c7c7c',
        weight='bold',
        fontsize=12,
        rotation=90
    )
    
    # ================================
    # 7. نص إحصائي على الملعب
    # ================================
    total_Shots = len(playerShots)
    Total_xg = round(playerShots['xG'].sum(), 2)
    n_goals = len(playerShots[playerShots['Outcome']=='Goal'])
    n_Ontarget = len(dataShotGood[dataShotGood['Player 1']==playerName])
    def safe_percent(part, total):
        return np.round((part / total) * 100, 2) if total > 0 else 0

    Shot_acc = safe_percent(n_Ontarget, total_Shots)
    
    ax_title = f'Total Shot: <{total_Shots}> \nGoal : <{n_goals}>\nShot On Target: <{n_Ontarget}> /{total_Shots} |{Shot_acc}% \nTotal xG: <{Total_xg}>'
    
    ax_text(
        58, 47, ax_title,
        ax=axs_Shots,
        highlight_textprops=[{"color": 'Black'}, {"color":"#56ae6c"}, {'color':'gold'}, {"color":"#7E1E9C"}],
        va='center', ha='center', fontsize=20
    )
    
    # ================================
    # 8. صورة خلفية (اختياري)
    # ================================
    ax_bg = fig_Shots.add_axes([0.75, .86, .15, .15])
    ax_bg.imshow(img, aspect='auto', alpha=1, zorder=-1)
    ax_bg.axis('off')
    pitch.inset_image(80.5, 40, img, height=40, alpha=.2, ax=axs_Shots, zorder=-1)
    
    # ================================
    # 10. سهم اتجاه الهجوم
    # ================================
    arrow = FancyArrowPatch(
        (0.12, 0.26), (0.12, 0.36),
        arrowstyle='-', linewidth=2,
        color='k', mutation_scale=20,
        transform=fig_Shots.transFigure, zorder=2
    )
    fig_Shots.patches.append(arrow)
    
    arrow1 = FancyArrowPatch(
        (0.12, 0.56), (0.12, 0.66),
        arrowstyle='->', linewidth=2,
        color='k', mutation_scale=20,
        transform=fig_Shots.transFigure, zorder=2
    )
    fig_Shots.patches.append(arrow1)
    
    fig_Shots.text(
        0.12, 0.41, "Attack Direction",
        ha='center', va='bottom',
        fontsize=18, color='k', zorder=3, rotation=90
    )
    
    # ================================
    # 11. إعداد Legend
    # ================================
    #marker_handles = []
    #for outcome, marker in markers.items():
     #   if outcome == "Goal":
     #      linestyle_ = linestyles.get(outcome, '-')
     #       marker_handles.append(
     #          pitch.scatter([], [], color='w', edgecolor='k', linestyle=linestyle_, marker=marker, s=300,  ax=axs_Shots)
     #       )
     #   else:
     #       linestyle_ = linestyles.get(outcome, '-')
     #       color = colors.get(outcome, 'none')
     #       edgecolor = edgecolors.get(outcome, 'none')
     #       marker_handles.append(
     #           pitch.scatter([], [], color=color, linestyle=linestyle_, marker=marker, s=300, edgecolor=edgecolor, ax=axs_Shots)
     #       )
    
    #fig_Shots.suptitle("Shots Map", fontsize=40, color='gold', y=.965, x=0.3)
    marker_handles = []
    for outcome, marker in markers.items():
        # لو الماركر ده كرة من mplsoccer، نستخدم 'o' للـ legend
        legend_marker = 'd' if marker == 'football' else marker
        if marker == 'football':
            face_color = 'g'
            edge_color = 'w'
        else:
            face_color = colors.get(outcome, 'white')
            edge_color = edgecolors.get(outcome, 'black')
    
        marker_handles.append(
            Line2D([0], [0],
                   marker=legend_marker,
                   color='w',  # الخط مش ظاهر
                   markerfacecolor=face_color,
                   markeredgecolor=edge_color,
                   markersize=12,
                   linestyle='None',
                   label=outcome)
        )

    axs_Shots.legend(handles=marker_handles,
                     loc='upper center',
                     bbox_to_anchor=(0.45, -0.05),
                     fontsize=12,
                     frameon=False,
                     labelspacing=1.2,
                     shadow=True,
                     ncol=1)

    #plt.legend(
    #    handles=marker_handles,
    #   labels=['Goal','Blocked','Off T','Saved','Wayward','Post'],
    #    loc='upper center',
    #    ncol=1,
    #   bbox_to_anchor=(-2.55, -4.62),
    #    labelspacing=1.2,
    #    fontsize=12,
    #    frameon=False,
    #   shadow=True
    #)

    fig_Shots.suptitle("Shots Map", fontsize=40, color='gold', y=.965, x=0.3)
    plt.show()

    #st.pyplot(fig_Shots)

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    
    # أبعاد المرمى
    pitch_width = 80
    
    goal_width = 8
    goal_height = 2.67
    depth = 0.6
    
    fig_Shots_on_frame, ax_Shots_on_frame = plt.subplots(figsize=(16,  14.25))
    
    # إحداثيات الشبكة الخلفية
    back_left = (depth, 0.25)
    back_right = (goal_width - depth, 0.25)
    back_top_left = (depth, goal_height)
    back_top_right = (goal_width - depth, goal_height)
    
    square_size = 0.18  # حجم المربعات
    
    # --------------------------------------------------------
    # ظل الشبكة الخلفية
    # --------------------------------------------------------
    ax_Shots_on_frame.fill_between(
        [back_left[0], back_right[0]],  # من x البداية للنهاية
        back_left[1],  # أسفل
        back_top_left[1],  # أعلى
        color='gray', alpha=0.1, zorder=0
    )
    
    # --------------------------------------------------------
    # الشبكة الخلفية
    # --------------------------------------------------------
    num_x = int(round((back_right[0] - back_left[0]) / square_size))
    num_y = int(round((back_top_left[1] - back_left[1]) / square_size))
    
    for i in range(num_x + 1):
        x = back_left[0] + i * (back_right[0] - back_left[0]) / num_x 
        ax_Shots_on_frame.plot([x, x], [back_left[1], back_top_left[1]], color='gray', lw=0.4, alpha=0.6)
    
    for j in range(num_y + 1):
        y = back_left[1] + j * (back_top_left[1] - back_left[1]) / num_y
        ax_Shots_on_frame.plot([back_left[0], back_right[0]], [y, y], color='gray', lw=0.4, alpha=0.6)
    
    # --------------------------------------------------------
    # الشبكة الجانبية اليسرى
    # --------------------------------------------------------
    num_x_side = int(round(depth / square_size))
    num_y_side = int(round((back_top_left[1] - back_left[1]) / square_size))
    
    for i in range(num_x_side + 1):
        x = i * (depth / num_x_side)
        y_start =  (back_left[1] ) * (i / num_x_side)
         # يبدأ عند back_left[1] وينخفض تدريجيًا
        y_end = back_top_left[1]  # النهاية عند أعلى الشبكة
    
    
        if i == num_x_side:
            lw = 2  # آخر خط
        else:
            lw = 0.5  # باقي الخطوط
        
        ax_Shots_on_frame.plot([x, x], [y_start, y_end], color='gray', lw=lw, alpha=0.8)
        # --------------------------------------------------------
    # ظل الشبكة الجانبية اليسرى
    # --------------------------------------------------------
        # حساب نقاط الظل وفق الميل
    x_shadow = []
    y_shadow = []
    
    for i in range(num_x_side + 1):
        x = i * (depth / num_x_side)
        y_start = (back_left[1]) * (i / num_x_side)  # نفس الميل السفلي للخطوط
        y_end = back_top_left[1]
        
        x_shadow.append(x)
        y_shadow.append(y_start)
    
    # نكمل الإحداثيات من أعلى الشبكة للعمق الأقصى عكس الاتجاه
    for i in reversed(range(num_x_side + 1)):
        x = i * (depth / num_x_side)
        y_shadow.append(back_top_left[1])
        x_shadow.append(x)
    
    # رسم الظل
    ax_Shots_on_frame.fill(x_shadow, y_shadow, color='r', alpha=0.08, zorder=0)
    
    
    # خطوط أفقية الشبكة الجانبية اليسرى
    for j in range(num_y_side + 1):
        y = back_left[1] + j * (back_top_left[1] - back_left[1]) / num_y_side
        ax_Shots_on_frame.plot([0, depth], [y, y], color='gray', lw=0.4, alpha=0.6)
    
    # --------------------------------------------------------
    # الشبكة الجانبية اليمنى
    # --------------------------------------------------------
    for i in range(num_x_side + 1):
        x = goal_width - depth + i * (depth / num_x_side)
        y_start = back_right[1] + i * (0 - back_right[1]) / num_x_side
        y_end = back_top_right[1] #- i * (back_top_right[1] - back_right[1]) / num_x_side
        
            # اجعل آخر خط أسمك
        if i == 0:
            lw = 2  # آخر خط
        else:
            lw = 0.5  # باقي الخطوط
        
        ax_Shots_on_frame.plot([x, x], [y_start, y_end], color='gray', lw=lw, alpha=0.8)
    
        # --------------------------------------------------------
    # ظل الشبكة الجانبية اليمنى
    # --------------------------------------------------------
        # حساب نقاط الظل وفق الميل
    x_shadow = []
    y_shadow = []
    
    for i in range(num_x_side + 1):
        x = goal_width - depth + i * (depth / num_x_side)
        y_start = back_right[1] + i * (0 - back_right[1]) / num_x_side  # نفس الميل السفلي للخطوط
        y_end = back_top_right[1]
        
        x_shadow.append(x)
        y_shadow.append(y_start)
    
    # نكمل الإحداثيات من أعلى الشبكة للعمق الأقصى عكس الاتجاه
    for i in reversed(range(num_x_side + 1)):
        x = goal_width - depth + i * (depth / num_x_side)
        y_shadow.append(back_top_right[1])
        x_shadow.append(x)
    
    # رسم الظل
    ax_Shots_on_frame.fill(x_shadow, y_shadow, color='r', alpha=0.08, zorder=0)
    
        
    # خطوط أفقية الشبكة الجانبية اليمنى
    for j in range(num_y_side + 1):
        y = back_right[1] + j * (back_top_right[1] - back_right[1]) / num_y_side
        ax_Shots_on_frame.plot([goal_width - depth, goal_width], [y, y], color='gray', lw=0.4, alpha=0.6)
    
    
    # --------------------------------------------------------
    # خطوط الزوايا (من القائمين للخلف)
    # --------------------------------------------------------
    ax_Shots_on_frame.plot([0, depth], [0, back_left[1]], color='gray', lw=3.5, alpha=1)
    ax_Shots_on_frame.plot([goal_width, goal_width - depth], [0, back_right[1]], color='gray', lw=3.5, alpha=1)
    ax_Shots_on_frame.plot([0+depth, goal_width-depth], [.25, .25], color='gray', lw=3.5, alpha=1)
    
    #ax.plot([0, depth], [goal_height, back_top_left[1]], color='g', lw=10, alpha=0.7)
    #ax.plot([goal_width, goal_width - depth], [goal_height, back_top_right[1]], color='g', lw=15, alpha=0.7)
    
    # --------------------------------------------------------
    # القائمين والعارضة
    # --------------------------------------------------------
    lw = 10
    goal_color = "#6F8FAF"
    ax_Shots_on_frame.plot([0, 0], [0, goal_height], color=goal_color, lw=lw, solid_capstyle='round', zorder=3)
    ax_Shots_on_frame.plot([goal_width, goal_width], [0, goal_height], color=goal_color, lw=lw, solid_capstyle='round', zorder=3)
    ax_Shots_on_frame.plot([0, goal_width], [goal_height, goal_height], color=goal_color, lw=lw, solid_capstyle='round', zorder=3)
    
    
    
    # --------------------------------------------------------
    # ظل الشبكة الجانبية اليمنى
    # --------------------------------------------------------
    ax_Shots_on_frame.fill_betweenx(
        [back_right[1], back_top_right[1]],
        goal_width - depth, goal_width,
        color='gray', alpha=0.08, zorder=0
    )
    
    # --------------------------------------------------------
    # خط المرمى بين القائمين
    # --------------------------------------------------------
    ax_Shots_on_frame.plot([0, goal_width], [-.035, -0.035], color='gray', lw=3, zorder=1)  # خط المرمى
    
    # ================================
    # 1. فلترة بيانات اللاعب
    # ================================
    # تحويل العمود من "x;y" إلى (x, y) كأرقام
    
    
    playerShots = dataShot[dataShot['Player 1'] == playerName].copy()
    
    def split_goal_location(val):
        if isinstance(val, str):
            val = val.replace(',', ';')  # توحيد الفاصل
            parts = val.split(';')
            if len(parts) == 2:
                try:
                    return [float(parts[0]), float(parts[1])]
                except ValueError:
                    return [np.nan, np.nan]
        elif isinstance(val, (tuple, list)) and len(val) == 2:
            return [val[0], val[1]]
        # ✅ لازم نرجع دايمًا list بطول 2
        return [np.nan, np.nan]
    
    # ✅ نفصل القيم باستخدام to_list() — الطريقة الأكثر أمانًا
    goal_xy = playerShots['Goal Location'].apply(split_goal_location).to_list()
    
    # تحويلها مباشرة إلى DataFrame بنفس الطول
    goal_xy_df = pd.DataFrame(goal_xy, columns=['Goal X', 'Goal Y'])
    
    # ✅ دمجها في DataFrame الأصلي بدون أخطاء
    playerShots = pd.concat([playerShots.reset_index(drop=True), goal_xy_df], axis=1)
    playerShots['Goal X' ]= playerShots['Goal X']*1.2-36
    playerShots['Goal Y' ]= playerShots['Goal Y']/100
    
    # ================================
    # تعريف الألوان لكل نوع قدم
    # ================================
    colors = {'Goal':'g','Blocked':'gray','Off T':'r','Saved':'gold','Wayward':'none','Post':'gold'}
    edgecolors = {'Goal':'w','Blocked':'k','Off T':'none','Saved':'gold','Wayward':'r','Post':'gold'}
    
    # تعريف الـ marker لكل Outcome
    markers = {'Goal':'d','Blocked':'s','Off T':'o','Saved':'o','Wayward':'o','Post':'d'}
    
    # تعريف linestyle لكل Outcome
    linestyles = {'Goal':'-','Blocked':'-','Off T':'--','Saved':'-','Wayward':'--','Post':'-'}
    
    # تعريف labels لكل Outcome
    labels = {'Goal':'Goal','Blocked':'Blocked','Off T':'Off T','Saved':'Saved','Wayward':'Wayward','Post':'Post'}
    
    
    
    
    ## ================================
    ## 4. رسم خطوط التمريرات
    ## ================================
    def draw_Shots_scatter(ax, playerShots):
        added_labels = set()  # لتجنب تكرار الـ legend
        
        for outcome, group in playerShots.groupby('Outcome'):
            for i, row in group.iterrows():
                if pd.isna(row['Goal X']) or pd.isna(row['Goal Y']):
                    continue  # تجاهل القيم الفارغة
                xg_value = row.get('xG', 0)
                mark = markers.get(outcome, 'o')
                label_text = labels.get(outcome, '-') if outcome not in added_labels else None
                linestyle_ = linestyles.get(outcome, '---')
                color = colors.get(outcome, 'none')
                edgecolor = edgecolors.get(outcome, 'none')
                
                if mark == 'd':
                    ax.scatter(
                        x=row['Goal X'],
                        y=row['Goal Y'],
                        s=xg_value*800,
                        c='g',
                        marker=mark,
                        edgecolor='w',
                        label=label_text,
                        lw=1,
                        zorder=4,
                       
                    )
                else:
                    ax.scatter(
                        x=row['Goal X'],
                        y=row['Goal Y'],
                        s=xg_value*800,
                        edgecolor=edgecolor,
                        marker=mark,
                        color=color,
                        label=label_text,
                        linestyle=linestyle_,
                        lw=1,
                        zorder=4,
                       
                    )
    
            added_labels.add(outcome)
    
    draw_Shots_scatter(ax_Shots_on_frame, playerShots)
    
    
    marker_handles = []
    for outcome, marker in markers.items():
        if outcome == "Goal":
            linestyle_ = linestyles.get(outcome, '-')
            marker_handles.append(
                ax_Shots_on_frame.scatter([], [], c='g', linestyle=linestyle_, marker=marker, s=300, edgecolor='w', )
            )
        else:
            linestyle_ = linestyles.get(outcome, '-')
            color = colors.get(outcome, 'none')
            edgecolor = edgecolors.get(outcome, 'none')
            marker_handles.append(
                ax_Shots_on_frame.scatter([], [], color=color, linestyle=linestyle_, marker=marker, s=300, edgecolor=edgecolor, )
            )
    
    
    plt.legend(
        handles=marker_handles,
        labels=['Goal','Blocked','Off T','Saved','Wayward','Post'],
        loc='upper center',
        ncol=6,
        bbox_to_anchor=(.44, 0.01),
        labelspacing=1.2,
        fontsize=12,
        frameon=False,
        shadow=True
    )
    
    
    ax_Shots_on_frame.scatter(        x=10+.3,
                        y=2.9+.5,
                        s=800,
                        edgecolor="r",
                        marker="o",
                        color="none",
                        linestyle="-",
                        lw=1,
                        zorder=5, )
    ax_Shots_on_frame.text(        x=9.95+.3,
                    y=2.84+.5,
                    s=len(playerShots),
                    weight="bold",
                    fontsize=14
    
           )
    
    ax_Shots_on_frame.text(        x=10.8+.3,
                    y=2.82+.5,
                    s="shots",
                    weight="bold",
                    color= "gray",
                    fontsize=20
    
           )
    ax_Shots_on_frame.scatter(        x=10+.3,
                        y=2.35+.5,
                        s=800,
                        edgecolor="r",
                        marker="o",
                        color="r",
                        alpha=.8,
                        linestyle="-",
                        lw=1,
                        zorder=5, )
    ax_Shots_on_frame.text(        x=9.95+.3,
                    y=2.3+.5,
                    s=len(playerShots[playerShots['Outcome']=="Goal"]),
                    fontsize=14, zorder=5,color="w",
                    weight="bold",alpha=1,
    
           )
    ax_Shots_on_frame.text(        x=10.8+.3,
                    y=2.3+.5,
                    s="goals",
                    fontsize=20,
                    weight="bold",
                    color= "gray"
    
           )
    ax_Shots_on_frame.text(        x=9.6+.5,
                    y=2.+.5,
                    s="_____"*3,
    
                    fontsize=25, zorder=4,color="gray",
                weight="bold",
    
           )
    ax_Shots_on_frame.text(        x=10.8+.5,
                    y=1.25+.5,
                    s="headers",
                    weight="bold",
                    fontsize=14, zorder=4,color="gray"
    
           )
    ax_Shots_on_frame.text(        x=11+.5,
                    y=1.45+.5,
                    s=len(playerShots[playerShots['Extra 2']=="Head"]),
    
                    fontsize=16,weight="bold", zorder=4,color="gray"
    
           )
    
    ax_Shots_on_frame.text(        x=10.8+.5,
                    y=.3+.5,
                    s="left foot",
                    weight="bold",
                    fontsize=14, zorder=4,color="gray"
    
           )
    ax_Shots_on_frame.text(        x=11.1+.5,
                    y=.5+.5,
                    s=len(playerShots[playerShots['Extra 2']=="Left Foot"]),
                    weight="bold",
                    fontsize=16, zorder=4,color="gray"
    
           )
    
    ax_Shots_on_frame.text(        x=10.8+.5,
                    y=-0.65+.5,
                    s="right foot",
                    weight="bold",
                    fontsize=14, zorder=4,color="gray"
    
           )
    
    ax_Shots_on_frame.text(        x=11.1+.5,
                    y=-.45+.5,
                    s=len(playerShots[playerShots['Extra 2']=="Right Foot"]),
                    weight="bold",
                    fontsize=16,  zorder=4,color="gray"
    
           )
    ax_Shots_on_frame.text(        x=10.95+.5,
                    y=-1.6+.5,
                    s="other",
                    weight="bold",
                    fontsize=14, zorder=4,color="gray"
    
           )
    
    ax_Shots_on_frame.text(        x=11.1+.5,
                    y=-1.4+.5,
                    s=len(playerShots[~playerShots['Extra 2'].isin(["Right Foot", "Left Foot", "Head"])]),
                    weight="bold",
                    fontsize=16,  zorder=4,color="gray"
    
           )
    
    fig_Shots_on_frame.text(x=0.5,y=0.2,s=" ", fontsize=40, color='gold')
    fig_Shots_on_frame.suptitle("Shots Map On Frame", fontsize=40, color='gold', y=.75, x=0.35)
    
    
    ax_bg = fig_Shots_on_frame.add_axes([0.95, .65, .15, .15])
    ax_bg.imshow(img, aspect='auto', alpha=1)
    ax_bg.axis('off')
    
    img_h = Image.open(r"Gemini_Generated_Image_panu9rpanu9rpanu (1).png") 
    
    ax_bg = fig_Shots_on_frame.add_axes([0.88, .44, .035, .035])
    ax_bg.imshow(img_h, aspect='auto', alpha=1)
    ax_bg.axis('off')
    
    img_l = Image.open(r"Gemini_Generated_Image_mh873gmh873gmh87.png") 
    
    ax_bg = fig_Shots_on_frame.add_axes([0.88, .37, .042, .042])
    ax_bg.imshow(img_l, aspect='auto', alpha=1)
    ax_bg.axis('off')
    
    img_r = Image.open(r"Gemini_Generated_Image_ji9npvji9npvji9n.png") 
    
    ax_bg = fig_Shots_on_frame.add_axes([0.88, .3, .042, .042])
    ax_bg.imshow(img_r, aspect='auto', alpha=1)
    ax_bg.axis('off')
    
    img_o = Image.open(r"Gemini_Generated_Image_os3vodos3vodos3v.png") 
    
    ax_bg = fig_Shots_on_frame.add_axes([0.88, .23, .042, .042])
    ax_bg.imshow(img_o, aspect='auto', alpha=1)
    ax_bg.axis('off')
    
    
    import numpy as np
    
    # ================================
    # إعداد شبكة 3x3 على المرمى
    # ================================
    x_edges = np.linspace(0, goal_width, 4)   # 0, 8/3, 16/3, 8
    y_edges = np.linspace(0, goal_height, 4)  # 0, goal_height/3, 2*goal_height/3, goal_height
    
    # ================================
    # حساب عدد التسديدات لكل مربع
    # ================================
    # استخدام np.histogram2d
    # فلترة القيم داخل حدود المرمى
    x = playerShots['Goal X'].dropna()
    y = playerShots['Goal Y'].dropna()
    
    # فلترة جميع القيم خارج المرمى
    valid_shots = playerShots[
        (playerShots['Goal X'] >= 0) & (playerShots['Goal X'] <= goal_width) &
        (playerShots['Goal Y'] >= 0) & (playerShots['Goal Y'] <= 1)
    ]
    
    x = valid_shots['Goal X']
    y = valid_shots['Goal Y']
    heatmap_counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    
    total_shots = heatmap_counts.sum
    # حساب مجموع التسديدات
    total_shots = heatmap_counts.sum()
    
    # حساب النسبة المئوية لكل مربع
    percent = (heatmap_counts / total_shots * 100) if total_shots > 0 else np.zeros_like(heatmap_counts)
    
    # رسم المربعات والنصوص
    for i in range(3):
        for j in range(3):
            # إحداثيات المربع
            x0 = x_edges[i]
            x1 = x_edges[i+1]
            y0 = y_edges[j]
            y1 = y_edges[j+1]
    
            # نسبة التسديدات في هذا المربع
            p = percent[i, j]
    
            # لون المربع حسب النسبة (colormap)
            color = plt.cm.Reds(p / 100)  # 0-1
    
            # رسم المربع باللون المناسب
            ax_Shots_on_frame.fill_between([x0, x1], y0, y1, color=color, alpha=0.6)
    
            # إضافة النص بالنسبة
            ax_Shots_on_frame.text(
                x=(x0+x1)/2,
                y=(y0+y1)/2,
                s=f"{int(p)}%",  # عرض النسبة
                ha='center',
                va='center',
                fontsize=14,
                color='black',
                weight='bold'
            )
    
    # ================================
    # رسم خطوط الشبكة لتقسيم المرمى
    # ================================
    # خطوط عمودية
    for xe in x_edges[1:-1]:
        ax_Shots_on_frame.plot([xe, xe], [0, goal_height], color='orange', lw=1.5,linestyle='--',alpha=1,zorder=1 )
    
    # خطوط أفقية
    for ye in y_edges[1:-1]:
        ax_Shots_on_frame.plot([0, goal_width], [ye, ye], color='orange', lw=1.5,linestyle='--',alpha=1,zorder=1 )
    
    # --------------------------------------------------------
    
    # الإعداد النهائي
    # --------------------------------------------------------
    ax_Shots_on_frame.set_xlim(-1, goal_width +2.5)
    ax_Shots_on_frame.set_ylim(-1, goal_height + 3)
    ax_Shots_on_frame.set_aspect('equal')
    ax_Shots_on_frame.axis('off')
    ax_Shots_on_frame.set_facecolor('#2E8B57')
    ax_Shots_on_frame.set_title("   ", fontsize=14)
    
    plt.show() 
    #st.pyplot(fig_Shots_on_frame)

    import streamlit as st
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image



# ===================== زر عرض التقرير =====================
#if st.button("عرض التقرير"):
#    # هنا ضع الكود اللي يعالج بيانات اللاعب ويرسم جميع الأشكال
#    # مثال على الشكل مع الصورة والخلفية
#    img = Image.open(r"WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg")
#    
#    fig, ax = plt.subplots(figsize=(16, 9))
#    ax.imshow(img)
#    ax.axis('off')
#    st.pyplot(fig)  # عرض الشكل على Streamlit
#    
# ===================== زر تحميل PDF =====================

# ===================== زر عرض التقرير =====================
# زر عرض التقرير

# ========= اختيار حجم العرض =========
# تقسيم الصفحة إلى عمودين
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64 
# إعداد الأعمدة
# 🧱 الأعمدة
# ================= اختيار حجم التقرير =================
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ===== تقسيم الصفحة إلى 3 أعمدة =====
col1, col2, col3 = st.columns([1, 2, 1])  # العمود الأوسط أوسع قليلاً

# ===== زر عرض التقرير =====
if st.button("عرض التقرير", key="show_report"):
    try:
        # نضع الكونتينر داخل العمود الأوسط فقط
        with col2:
            st.title(f"📑 {selected_player}")

            # ===== غلاف التقرير =====
            img = Image.open("WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg")
            fig_cover, ax_cover = plt.subplots(figsize=(16, 9))
            ax_cover.imshow(img)
            ax_cover.axis('off')
            st.pyplot(fig_cover)

            st.markdown("---")

            # ===== التبويبات =====
            tab1, tab2, tab3 = st.tabs(["🎯 تمريرات", "⚽ تسديدات", "📊 إحصائيات"])

            with tab1:
                st.subheader("🎯 تمريرات اللاعب")
                try:
                    st.pyplot(fig_PassTable)
                    st.pyplot(fig_passes)
                except:
                    st.info("🚫 لا توجد بيانات للتمريرات.")

            with tab2:
                st.subheader("⚽ تسديدات اللاعب")
                try:
                    st.pyplot(fig_ShotTable)
                    st.pyplot(fig_Shots)
                    st.pyplot(fig_Shots_on_frame)
                except:
                    st.info("🚫 لا توجد بيانات للتسديدات.")

            with tab3:
                st.subheader("📊 إحصائيات عامة")
                try:
                    st.dataframe(PassesTable)
                except:
                    st.info("🚫 لا توجد إحصائيات عامة.")

        st.success("✅ تم عرض التقرير بنجاح.")

    except Exception as e:
        st.error(f"حدث خطأ أثناء عرض التقرير: {e}")
# ===================== زر تحميل PDF =====================
if st.button("تحميل التقرير PDF"):
    # إنشاء نسخة PDF للتقرير
    img = Image.open(r"WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg")
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    ax1.imshow(img)
    ax1.axis('off')

    pdf_path = f"PATA_STATS_Player_Report_{playerName}.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig_PassTable)
        pdf.savefig(fig_passes)
        pdf.savefig(fig_ShotTable)
        pdf.savefig(fig_Shots)
        pdf.savefig(fig_Shots_on_frame, bbox_inches='tight', pad_inches=.25)

    plt.close('all')

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="⬇️ تحميل التقرير PDF",
            data=f,
            file_name=f"PATA_STATS_{playerName}_Report.pdf",
            mime="application/pdf"
        )
