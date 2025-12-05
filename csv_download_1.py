

import streamlit as st
import json
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import bcrypt

# إعداد الصفحة
st.set_page_config(
    page_title="PATA STATS - Login",
    layout="centered",
    page_icon=r"C:\Users\Mo\Downloads\WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg"
)

USERS_FILE = "users.json"
ADMIN_EMAIL = "pata.stats10@gmail.com"       # ضع ايميلك هنا
ADMIN_PASSWORD = "tftd wrwt vhyi wruf"       # App Password من Google
ADMIN_PANEL_URL = "http://localhost:8501/?admin=true" # عدل اللينك حسب السيرفر بتاعك
# تحميل المستخدمين
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# حفظ المستخدمين
def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

# إرسال إيميل إشعار
def send_email(new_username, new_password):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEText(
        f"📢 طلب تسجيل جديد\n\n"
        f"👤 اسم المستخدم: {new_username}\n"
        f"🔑 كلمة المرور: {new_password}\n"
        f"📅 تاريخ التسجيل: {now}\n\n"
        f"اضغط هنا لإدارة الحسابات:\n{ADMIN_PANEL_URL}"
    )
    msg["Subject"] = f"طلب تسجيل جديد - {new_username}"
    msg["From"] = ADMIN_EMAIL
    msg["To"] = ADMIN_EMAIL

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(ADMIN_EMAIL, ADMIN_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.error(f"Error sending email: {e}")

# ---------------- صفحات ----------------

# تسجيل جديد
def signup():
    st.title("📝 Create New Account")
    new_username = st.text_input("Username", key="signup_username")
    new_password = st.text_input("Password", type="password", key="signup_password")
    if st.button("Sign up"):
        users = load_users()
        if new_username in users:
            st.error("❌ اسم المستخدم موجود بالفعل.")
        elif not new_username or not new_password:
            st.error("⚠️ Please enter both username and password.")
        else:
            expiry_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            hashed_pw = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            users[new_username] = {
                "password": hashed_pw,
                "status": "pending",
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "expiry_date": expiry_date
            }
            save_users(users)
            send_email(new_username, new_password)
            st.info("⏳ Account created and pending approval by admin.")
    if st.button("🔙 رجوع لصفحة تسجيل الدخول"):
        
        st.session_state["page"] = "login"
        st.rerun()
    
# تسجيل الدخول
def login():
    st.title("🔑 Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        users = load_users()
        if username in users:
            if users[username]["status"] == "approved":
                stored_hash = users[username]["password"].encode("utf-8")
                if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("❌ كلمة المرور غير صحيحة.")
            elif users[username]["status"] == "pending":
                st.warning("⏳ حسابك في انتظار موافقة الأدمن.")
            else:
                st.error("🚫 تم رفض الحساب من قبل الأدمن.")
        else:
            st.error("❌ اسم المستخدم غير موجود.")

    st.write("---")
    # زر يفتح صفحة التسجيل
    if st.button("📝 Create New Account"):
        st.session_state["page"] = "signup"
        st.rerun() # لازم تعمل ملف اسمه signup.py في مجلد pages


    # 🔹 معلومات الاشتراك بلوكات تحت بعض
    with st.container():
        st.markdown(
            """
            <div style="background-color:#2C3E50; padding:15px; border-radius:10px; margin-bottom:10px;">
                💰 سعر الاشتراك: <b>500 جنيه</b>
                <h4>🎁 Free Trial
                For 1 Month </h4>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.container():
        st.markdown(
           """
            <div style="background-color:#145A32; padding:15px; border-radius:10px; margin-bottom:10px; color:white;">
                 📧 Email: <a href="mailto:pata.stats10@gmail.com" style="color:#FFD700; text-decoration:none;">
                    pata.stats10@gmail.com
                </a><br>
                 ⏳ تصلك النسخة المجانية بعد إرسال الطلب من خلال البريد<br>
                <a href="https://wa.me/201558155922" target="_blank">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" 
                         width="25" height="25" style="margin:10px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )



  

# لوحة الأدمن
def admin_panel():
    st.title("⚙️ Admin Panel")

    users = load_users()
    if not users:
        st.info("لا يوجد مستخدمين حالياً.")
        return

    # اختيار يوزر من القائمة المنسدلة
    selected_user = st.selectbox("👥 اختر المستخدم:", list(users.keys()))

    if selected_user:
        user_data = users[selected_user]

        # ✅ عرض بيانات المستخدم
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#2C3E50; color:white; padding:20px; border-radius:10px; margin-bottom:20px;">
                    <h3>👤 {selected_user}</h3>
                    <p>📌 الحالة: <b>{user_data.get("status", "N/A")}</b></p>
                    <p>📅 تاريخ الإنشاء: <b>{user_data.get("created_at", "N/A")}</b></p>
                    <p>⏳ تاريخ انتهاء الاشتراك: <b>{user_data.get("expiry_date", "N/A")}</b></p>
                    <p>🔑 كلمة السر (Hashed): <small>{user_data.get("password", "N/A")}</small></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ✅ أزرار التحكم
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("✅ Approve"):
                users[selected_user]["status"] = "approved"
                save_users(users)
                st.success(f"تم تفعيل {selected_user}")
                st.rerun()

        with col2:
            if st.button("❌ Reject"):
                users[selected_user]["status"] = "rejected"
                save_users(users)
                st.warning(f"تم رفض {selected_user}")
                st.rerun()

        with col3:
            new_date = st.date_input("📅 اختر تاريخ جديد للاشتراك")
            if st.button("🔄 تجديد الاشتراك"):
                users[selected_user]["expiry_date"] = new_date.strftime("%Y-%m-%d")
                save_users(users)
                st.success(f"✅ تم تجديد اشتراك {selected_user} حتى {new_date}")
                st.rerun()

        with col4:
            if st.button("🔑 Reset Password (00000000)"):
                new_pass = "00000000"
                hashed_pw = bcrypt.hashpw(new_pass.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                users[selected_user]["password"] = hashed_pw
                save_users(users)
                st.success(f"🔄 تم إعادة تعيين كلمة مرور {selected_user} إلى: {new_pass}")

# القائمة الرئيسية بعد تسجيل الدخول

def main_menu():
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
    from mplsoccer import Pitch
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
    
    from mplsoccer import VerticalPitch
    import matplotlib.pyplot as plt
    from io import BytesIO
    from matplotlib.ticker import StrMethodFormatter
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    username = st.session_state["username"]

    # 🟢 Sidebar: ترحيب + تسجيل الخروج
    st.sidebar.title(f"👋 مرحباً {username}")

    # 🟢 عرض بيانات اللاعب
    #t.subheader(f"📊 بيانات اللاعب: {username}")
    #ry:
    #   if hasattr(csv_download_1, "show"):
    #       csv_download_1.show(username)
    #   else:
    #       st.warning("⚠️ لم يتم العثور على دالة عرض بيانات اللاعب.")
    #xcept Exception:
    #   st.markdown(
    #       """
    #       <div style="background-color:#FDEDEC; 
    #                   border:2px solid #E74C3C; 
    #                   border-radius:10px; 
    #                   padding:20px; 
    #                   text-align:center;">
    #           <h2 style="color:#E74C3C;">🚫 لا توجد بيانات متاحة لهذا اللاعب حالياً</h2>
    #           <p>يرجى التواصل مع الأدمن لإضافة بياناتك.</p>
    #       </div>
    #       """,
    #       unsafe_allow_html=True
    #   )
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
    halfs = ['1st Half', '2nd Half']
    
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    def detect_start_side(data, start_side):
        halfs = ['1st Half', '2nd Half']
    
        # ==================================================
        # 1️⃣ تحديد الاتجاه من الحارس (لو موجود)
        # ==================================================
        goalkeeper_events = data[data['Event'] == 'Goal Keeper']
    
        if not goalkeeper_events.empty:
            first_gk_x = goalkeeper_events['Actions positions x'].iloc[0]
            if first_gk_x < 60:
                halfsToChangeXY = [halfs[1]]
                st.success("✅  الفريق بدأ من اليسار — سيتم قلب الشوط الثاني بناء على الحارس.")
            else:
                halfsToChangeXY = [halfs[0]]
                st.success("✅ الفريق بدأ من اليمين — سيتم قلب الشوط الأول. بناء على الحارس")
        else:
            # ==================================================
            # 2️⃣ الاتجاه اليدوي كخطة بديلة
            # ==================================================
            st.warning("⚠️ لا يوجد حدث لحارس المرمى، سيتم استخدام الاتجاه اليدوي.")
    
            if start_side.lower() == "left":
                halfsToChangeXY = [halfs[1]]
                st.info("➡️ الفريق بدأ من اليسار — سيتم قلب الشوط الثاني.")
            elif start_side.lower() == "right":
                halfsToChangeXY = [halfs[0]]
                st.info("➡️ الفريق بدأ من اليمين — سيتم قلب الشوط الأول.")
            else:
                halfsToChangeXY = [halfs[1]]
                st.error("⚠️ اتجاه غير معروف، تم استخدام الافتراضي (left): سيتم قلب الشوط الثاني.")
    
        # ==================================================
        # 3️⃣ قلب الاتجاهات
        # ==================================================
        def flip_coordinates(row):
            if row['Half'] in halfsToChangeXY:
                row['Actions positions x'] = 120 - row['Actions positions x']
                row['Actions positions y'] = 80 - row['Actions positions y']
                row['Actions positions x End'] = 120 - row['Actions positions x End']
                row['Actions positions y End'] = 80 - row['Actions positions y End']
            return row
    
        data = data.apply(flip_coordinates, axis=1)
    
        st.success("✅ تم تحديث الاتجاهات بنجاح.")
        return data
    # ================= Streamlit Config =================
    #st.set_page_config(page_title="Dynamic Player Analysis", layout="wide")
    st.title("📊 رفع CSV وتحليلات اللاعبين ديناميكية")
    
    start_side = st.radio(
            "اختر الاتجاه الذي بدأ منه الفريق:",
            options=["left", "right"],
            index=0,
            horizontal=True
        )
    
    # ================= File Upload =================
    uploaded_file = st.file_uploader("اختر ملف CSV ", type=["csv"])
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
    
        ##st.write(f"✅ الفريق بدأ من: **{start_side.upper()}**")
    
    
    
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
           dataThroughBall,
            dataError
        )= data_pre_procces(uploaded_file,start_side)
        st.dataframe(data.head())
        df_1 = detect_start_side(data, start_side) ##
        st.dataframe(df_1) ##
    
    # ================== Player Avg Positions & Total Actions =================
    
        playersNames = sorted(data["Player 1"].dropna().unique().tolist())
        # نضيف خيار الفريق بالكامل في أول القائمة
        playersNames = ["Team"] + playersNames
        
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
    
        # ✳️ دالة موحدة للفلترة — تدعم "Team"
        def get_player_data(df, playerName):
            if playerName == "Team":
                return df.copy()
            return df[df['Player 1'] == playerName]
        
        
        # فلترة البيانات حسب اللاعب أو الفريق كله
        player_data = get_player_data(data, selected_player)
        
        st.subheader(f"📋 بيانات    {selected_player}")
        st.dataframe(player_data[player_data['Half']==halfs[0]])
        st.dataframe(player_data[player_data['Half']==halfs[1]])
        playerName = selected_player
    
        
        playerNumberOfActions = []
        playerData = get_player_data(actionData, playerName)
        playerActionBadData = get_player_data(actionBadData, playerName)
        playerActionGoodData = get_player_data(actionGoodData, playerName)
        
    
    
        def generate_passes_table(playerName, dataPass, dataPassGood, dataAsist, dataSecondAsist,
                              dataKeyPass, dataprogpass, dataprogpassGood,
                              dataThroughBall, dataCross, dataCrossGood, halfs):
            PassesTable = pd.DataFrame()
            
            """ Passes / accurate Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                actions1=get_player_data(dataPass, playerName),
                actions1Good=get_player_data(dataPassGood, playerName),
                rowName='    Passes / accurate', nSpaces=12
            )
            
            """ Assist / accurate Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                get_player_data(dataAsist, playerName),
                '-',
                '    Assist', 12
            )
            
            """ 2nd Assist / accurate Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                get_player_data(dataSecondAsist, playerName),
                '-',
                '    2nd Assist', 12
            )
            
            """ key / accurate Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                get_player_data(dataKeyPass, playerName),
                '-',
                '    Key pass', 12
            )
            
            """ progressive PASSES / accurate Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                actions1=get_player_data(dataprogpass, playerName),
                actions1Good=get_player_data(dataprogpassGood, playerName),
                rowName='    progressive Passes / accurate', nSpaces=12
            )
            
            """ progressive PASSES / into the Final 3rd Both Halfs """
            dP = get_player_data(dataprogpass, playerName)
            dPG = get_player_data(dataprogpassGood, playerName)
            mask = (dP['Actions positions x End'] >= 80)
            maskG = (dPG['Actions positions x End'] >= 80)
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName=' prg pass into the Att 3rd / accurate', nSpaces=12
            )
            
            """ progressive PASSES / into the box Both Halfs """
            mask = (dP['Actions positions x End'] >= 102) & (dP['Actions positions y End'].between(22, 58))
            maskG = (dPG['Actions positions x End'] >= 102) & (dPG['Actions positions y End'].between(22, 58))
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='    prg pass into the box / accurate', nSpaces=12
            )
            
            """ Throught Pass Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                actions1=get_player_data(dataThroughBall, playerName),
                actions1Good='-',
                rowName='    Through Pass', nSpaces=12
            )
            
            """ PASSES / into the box Both Halfs """
            dP = get_player_data(dataPass, playerName)
            dPG = get_player_data(dataPassGood, playerName)
            mask = (dP['Actions positions x End'] >= 102) & (dP['Actions positions y End'].between(22, 58))
            maskG = (dPG['Actions positions x End'] >= 102) & (dPG['Actions positions y End'].between(22, 58))
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='    into the box / accurate', nSpaces=12
            )
            
            """ PASSES / into the Final 3rd Both Halfs """
            mask = (dP['Actions positions x End'] >= 80)
            maskG = (dPG['Actions positions x End'] >= 80)
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='    into the Att 3rd / accurate', nSpaces=12
            )
            
            """ Cross / Accurate Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                actions1=get_player_data(dataCross, playerName),
                actions1Good=get_player_data(dataCrossGood, playerName),
                rowName='    Cross / accurate', nSpaces=12
            )
            
            """ Cross / into the box Both Halfs """
            dP = get_player_data(dataCross, playerName)
            dPG = get_player_data(dataCrossGood, playerName)
            mask = (dP['Actions positions x End'] >= 102) & (dP['Actions positions y End'].between(22, 58))
            maskG = (dPG['Actions positions x End'] >= 102) & (dPG['Actions positions y End'].between(22, 58))
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='   Cross into the box / accurate', nSpaces=12
            )
            
            """ PASSES / Ground Both Halfs """
            dPg = get_player_data(dataPass, playerName)
            dPgG = get_player_data(dataPassGood, playerName)
            PassesTable = addTableRow(
                PassesTable,
                actions1=dPg[dPg['Event']=='Ground Pass'],
                actions1Good=dPgG[dPgG['Event']=='Ground Pass'],
                rowName='    Ground Pass / accurate', nSpaces=12
            )
            
            """ PASSES / Low Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                actions1=dPg[dPg['Event']=='Low Pass'],
                actions1Good=dPgG[dPgG['Event']=='Low Pass'],
                rowName='    Low Pass / accurate', nSpaces=12
            )
            
            """ PASSES / High Both Halfs """
            PassesTable = addTableRow(
                PassesTable,
                actions1=dPg[dPg['Event']=='High Pass'],
                actions1Good=dPgG[dPgG['Event']=='High Pass'],
                rowName='    High Pass / accurate', nSpaces=12
            )
            
            """ PASSES / Forward Both Halfs """
            dP = get_player_data(dataPass, playerName)
            dPG = get_player_data(dataPassGood, playerName)
            mask = dP['Actions positions x'] < dP['Actions positions x End']
            maskG = dPG['Actions positions x'] < dPG['Actions positions x End']
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='    forward / accurate', nSpaces=12
            )
            
            """ PASSES / Back Both Halfs """
            mask = dP['Actions positions x'] >= dP['Actions positions x End']
            maskG = dPG['Actions positions x'] >= dPG['Actions positions x End']
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='    back / accurate', nSpaces=12
            )
            
            """ PASSES / to the right Both Halfs """
            mask = dP['Actions positions y'] < dP['Actions positions y End']
            maskG = dPG['Actions positions y'] < dPG['Actions positions y End']
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='    to the right / accurate', nSpaces=12
            )
            
            """ PASSES / to the left Both Halfs """
            mask = dP['Actions positions y'] >= dP['Actions positions y End']
            maskG = dPG['Actions positions y'] >= dPG['Actions positions y End']
            PassesTable = addTableRow(
                PassesTable,
                actions1=dP[mask],
                actions1Good=dPG[maskG],
                rowName='    to the left / accurate', nSpaces=12
            )
            
            """ Average Pass length Both Halfs """
            dPH1 = round(dP[dP['Half'] == halfs[0]]['Actions Pos Length'].mean(), 1)
            dPH2 = round(dP[dP['Half'] == halfs[1]]['Actions Pos Length'].mean(), 1)
            
            print1 = pd.DataFrame({
                'PATASTATS INDEX': '    Average Pass length',
                'Per Match': round(dP['Actions Pos Length'].mean(), 1),
                '1st half': dPH1,
                '2nd half': dPH2,
            }, index=[0])
            PassesTable = pd.concat([PassesTable, print1], ignore_index=True)
            
            return PassesTable
    
        
        player_passes_table = generate_passes_table(playerName, dataPass, dataPassGood, dataAsist,
                                                dataSecondAsist, dataKeyPass, dataprogpass,
                                                dataprogpassGood, dataThroughBall, dataCross,
                                                dataCrossGood, halfs)
    
    
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
        PassesTable = convert_percent_columns(player_passes_table)
    
        
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
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patheffects as path_effects
        from mplsoccer import VerticalPitch
        import matplotlib.pyplot as plt
        from io import BytesIO
        from matplotlib.ticker import StrMethodFormatter
        from matplotlib.patches import FancyArrowPatch
    
        # Colormap شبيه بـ StatsBomb
        statsbomb_cmap_red_blue = LinearSegmentedColormap.from_list(
            "Blue-Gray-Red",
            ['#000B2B',  # أزرق غامق
             '#929591',  # رمادي
             '#8C000F'], # أحمر غامق
            N=20
        )
            
        
        def generate_player_heatmap(playerName, actionGoodData, actionBadData, playerData, img):
            """
            دالة لرسم Heatmap لأفعال اللاعب مع تمييز الأفعال الجيدة والسيئة
            وإضافة الصور والشعارات والسهم الخاص باتجاه الهجوم.
            
            Parameters:
            -----------
            playerName : str
                اسم اللاعب
            actionGoodData : pd.DataFrame
                بيانات الأفعال الجيدة
            actionBadData : pd.DataFrame
                بيانات الأفعال السيئة
            playerData : pd.DataFrame
                بيانات جميع الأفعال للاعب
            img : np.array or PIL.Image
                الصورة/الشعار لإضافته على الملعب
            
            Returns:
            --------
            BytesIO
                الصورة الناتجة بصيغة PNG
            """
            
    
            # فلترة البيانات
            actionsPlayerGood  = get_player_data(actionGoodData, playerName)
            actionsPlayerBad   = get_player_data(actionBadData, playerName)
            # إنشاء الشكل
            fig, ax = plt.subplots(figsize=(16, 9))
        
            # إعداد الملعب الرأسي
            pitch = Pitch(pitch_color='w', line_color='k', line_zorder=2)
            pitch.draw(ax=ax)
        
            # رسم الأفعال الجيدة
            pitch.scatter(
                actionsPlayerGood['Actions positions x'],
                actionsPlayerGood['Actions positions y'],
                c='lime', s=80, ax=ax, label='Good Actions', zorder=2
            )
        
            # رسم الأفعال السيئة
            pitch.scatter(
                actionsPlayerBad['Actions positions x'],
                actionsPlayerBad['Actions positions y'],
                c='red', s=80, marker='x', ax=ax, label='Bad Actions', zorder=2
            )
        
            # إعداد path_effects للنصوص على Heatmap
            path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                        path_effects.Normal()]
        
            # إنشاء Heatmap
            bin_statistic = pitch.bin_statistic(
                playerData['Actions positions x'],
                playerData['Actions positions y'],
                statistic='count',
                normalize=True,
                bins=(6, 4)
            )
            
            hm = pitch.heatmap_positional([bin_statistic], ax=ax,
                                          cmap=statsbomb_cmap_red_blue, edgecolors='None', zorder=1, alpha=.7)
            
            # إضافة النصوص على Heatmap
            pitch.label_heatmap(
                bin_statistic, color='#f4edf0', fontsize=12,
                ax=ax, ha='center', va='center',
                str_format='{:.0%}', path_effects=path_eff
            )
        
            # إضافة colorbar
            cax = fig.add_axes([0.62, 0.05, 0.15, 0.01])
            cbar = fig.colorbar(hm[0], cax=cax, ax=ax, fraction=0.016, pad=0.03, orientation='horizontal',
                                format=StrMethodFormatter("{x:.0%}"))
            cbar.set_label('Action Intensity', fontsize=14)
        
            # إضافة الصور/الشعارات على الملعب
            pitch.inset_image(60.5, 45, img, height=70, alpha=.2, ax=ax, zorder=-1)
            pitch.inset_image(115, -15, img, height=25, alpha=1, ax=ax, zorder=1)
        
            # العنوان
            ax.set_title(f"\n\nHeatmap of All Actions", fontsize=35, color='gold', y=1.0, x=.4)
        
            # رسم السهم للاتجاه الهجومي
            arrow = FancyArrowPatch((40, 82), (80, 82), arrowstyle='->', linewidth=1, color='k', mutation_scale=10, zorder=2)
            ax.add_patch(arrow)
        
            ax.annotate(
                'Attack Direction',
                xy=(60, 85),
                ha='center',
                color='k',
                fontsize=14,
                zorder=2
            )
        
            # وسوم legend
            legend = ax.legend(title='', labelspacing=2, loc="upper center", ncol=2,
                               frameon=False, fancybox=True, shadow=True,
                               bbox_to_anchor=(0.2, -0.06), markerscale=1.5, title_fontsize=34)
            
            for text in legend.get_texts():
                text.set_fontsize(14)
                text.set_color('k')
        
            # ضبط حدود الملعب
            ax.set_xlim(-10, 130)
            ax.set_ylim(85, -15)
        
            # حفظ الصورة في BytesIO
            playerActionsImage = BytesIO()
            fig.savefig(playerActionsImage, format='png', edgecolor='white', bbox_inches='tight', pad_inches=0, transparent=False)
            playerActionsImage.seek(0)
            plt.close(fig)
        
            return fig
        fig = generate_player_heatmap(playerName, actionGoodData, actionBadData, playerData, img)
    
    
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
        from mplsoccer import VerticalPitch
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.ticker import StrMethodFormatter
        from highlight_text import ax_text
    
        def filter_half(data, half):
            return data[data['Half'] == half]
        
    
        def plot_player_pass_map(playerName, dataPassGood, dataPassBad, dataKeyPass,
                             dataAsist, dataSecondAsist, statsbomb_cmap_red_blue, img):
    
            playerPassGood     =  get_player_data(dataPassGood, playerName)
            playerPassBad      =  get_player_data(dataPassBad, playerName)
            playerDataKeyPass  =  get_player_data(dataKeyPass, playerName)
            playerAssist       =  get_player_data(dataAsist, playerName)
            player2ndAssist    =  get_player_data(dataSecondAsist, playerName)
            
            halves = ['1st Half', '2nd Half']
            
    
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
            
                passes = {
                    'Completed passes': (pass_good, '#56ae6c'),
                    'Incomplete passes': (pass_bad, '#ba4f45'),
                    'Key passes': (key_pass, '#7E1E9C'),
                    'Assist': (assist, 'gold'),
                    '2nd Assist': (second_assist, 'orange')
                }
            
                def draw_type(df, color, label):
                    if df is None or df.empty:
                        return
                    pitch.lines(
                        df['Actions positions x'], df['Actions positions y'],
                        df['Actions positions x End'], df['Actions positions y End'],
                        lw=3, transparent=True, comet=True,
                        color=color, zorder=3, ax=ax, label=label
                    )
                    pitch.scatter(
                        df['Actions positions x End'], df['Actions positions y End'],
                        ax=ax, edgecolor="w", facecolor=color,
                        s=30, lw=.5, alpha=0.9, zorder=4
                    )
            
                for label, (df, color) in passes.items():
                    draw_type(df, color, label)
        
            for i, data in enumerate(half_data):
                draw_pass_lines(axs_passes[i], *data)
                axs_passes[i].set_title(f"{halves[i]}", y=1.05, fontsize=22, color='k')
            
            # ================================
            # 5. دالة Heatmap جديدة (أفقي وعمودي)
            # ================================
            def draw_heatmap_full(ax, data, pitch, cmap):
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
            cbar = fig_passes.colorbar(heatmaps[0][0], cax=cax,  location="left", fraction=0.016, pad=0.0,
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
            
            return fig_passes
    
        fig_passes = plot_player_pass_map(playerName, dataPassGood, dataPassBad, dataKeyPass,
                                      dataAsist, dataSecondAsist, statsbomb_cmap_red_blue, img)
        
    
        def create_shooting_table(playerName, dataShot, dataShotGood, halfs):
            shootingTable = DataFrame()
            
            # فلترة البيانات لكل لاعب
            playerShot     = get_player_data(dataShot, playerName)
            playerShotGood = get_player_data(dataShotGood, playerName)
            
            # أقنعة التسديدات داخل وخارج منطقة الجزاء
            shotInBoxMask = (playerShot['Actions positions x']>=102) & (playerShot['Actions positions y']>=22) & (playerShot['Actions positions y']<=80-22)
            shotInBoxMaskGood = (playerShotGood['Actions positions x']>=102) & (playerShotGood['Actions positions y']>=22) & (playerShotGood['Actions positions y']<=80-22)
            
            # ركلات حرة وركلات جزاء
            shotsfromFreekick = playerShot[playerShot['Extra 1']=="Free Kick"]
            shotsfromFreekickGood = playerShotGood[playerShotGood['Extra 1']=="Free Kick"]
            
            PenaltyShot = playerShot[playerShot['Extra 1']=="Penalty"]
            PenaltyShotGood = PenaltyShot[PenaltyShot['Outcome']=='Goal']
            
            # =========================
            # إضافة الصفوف إلى الجدول
            # =========================
            shootingTable = addTableRow(shootingTable, playerShot[playerShot['Outcome']=="Goal"], '-', '    Goals', 12)
            shootingTable = addTableRow(shootingTable, playerShot, playerShotGood, '    Shots / on Target', 12)
            shootingTable = addTableRow(shootingTable, playerShot[shotInBoxMask], playerShotGood[shotInBoxMaskGood], '    Shots in Box / on Target', 12)
            shootingTable = addTableRow(shootingTable, playerShot[~shotInBoxMask], playerShotGood[~shotInBoxMaskGood], '    Shots Out OF Box / on Target', 12)
            shootingTable = addTableRow(shootingTable, shotsfromFreekick, shotsfromFreekickGood, '    Free Kicks Shots / on Target', 12)
            shootingTable = addTableRow(shootingTable, PenaltyShot, PenaltyShotGood, '    Penalty / Goal', 12)
            
            shootingTable = addTableRow(shootingTable, playerShotGood, playerShotGood[playerShotGood['Extra 2']=='Head'], '    Shots on Target / Head', 12)
            shootingTable = addTableRow(shootingTable, playerShotGood, playerShotGood[playerShotGood['Extra 2']=='Right Foot'], '    Shots on Target / R Foot', 12)
            shootingTable = addTableRow(shootingTable, playerShotGood, playerShotGood[playerShotGood['Extra 2']=='Left Foot'], '    Shots on Target / L Foot', 12)
            
            # =========================
            # Goal conversion ratio
            # =========================
            dP_total = len(playerShot[playerShot['Outcome'] == "Goal"]) / max(1,len(playerShot))
            dP_H1 = len(playerShot[(playerShot['Outcome'] == "Goal") & (playerShot['Half'] == halfs[0])]) / max(1,len(playerShot[playerShot['Half'] == halfs[0]]))
            dP_H2 = len(playerShot[(playerShot['Outcome'] == "Goal") & (playerShot['Half'] == halfs[1])]) / max(1,len(playerShot[playerShot['Half'] == halfs[1]]))
            
            shootingTable = pd.concat([shootingTable, pd.DataFrame({
                'PATASTATS INDEX': ['    Goal Conversion Ratio'],
                'Per Match': [round(dP_total, 2)],
                '1st half': [round(dP_H1, 2)],
                '2nd half': [round(dP_H2, 2)],
            })], ignore_index=True)
            
            # =========================
            # XG
            # =========================
            for label, data_filter in [('XG', playerShot), ('XGoT', playerShotGood)]:
                per_match = round(data_filter['xG'].sum(), 2)
                h1 = round(data_filter[data_filter['Half']==halfs[0]]['xG'].sum(),2)
                h2 = round(data_filter[data_filter['Half']==halfs[1]]['xG'].sum(),2)
                shootingTable = pd.concat([shootingTable, pd.DataFrame({
                    'PATASTATS INDEX': [f'    {label} '],
                    'Per Match': [per_match],
                    '1st half': [h1],
                    '2nd half': [h2]
                })], ignore_index=True)
            
            # np XG
            dP_np = playerShot[playerShot['Extra 1']!="Penalty"]
            per_match = round(dP_np['xG'].sum(),2)
            h1 = round(dP_np[dP_np['Half']==halfs[0]]['xG'].sum(),2)
            h2 = round(dP_np[dP_np['Half']==halfs[1]]['xG'].sum(),2)
            shootingTable = pd.concat([shootingTable, pd.DataFrame({
                'PATASTATS INDEX': ['    np XG  '],
                'Per Match': [per_match],
                '1st half': [h1],
                '2nd half': [h2]
            })], ignore_index=True)
            
            # XGoT by parts (Head, R Foot, L Foot)
            for part in ['Head','Right Foot','Left Foot']:
                dP = playerShotGood[playerShotGood['Extra 2']==part]
                per_match = round(dP['xG'].sum(),2)
                h1 = round(dP[dP['Half']==halfs[0]]['xG'].sum(),2)
                h2 = round(dP[dP['Half']==halfs[1]]['xG'].sum(),2)
                shootingTable = pd.concat([shootingTable, pd.DataFrame({
                    'PATASTATS INDEX': [f'    XGoT By {part} '],
                    'Per Match': [per_match],
                    '1st half': [h1],
                    '2nd half': [h2]
                })], ignore_index=True)
            
            # np XG per shot
            for label, dP in [('np XG per shot', playerShot[playerShot['Extra 1']!="Penalty"])]:
                per_match = round(dP['xG'].sum()/max(1,len(dP)),2)
                h1 = round(dP[dP['Half']==halfs[0]]['xG'].sum()/max(1,len(dP[dP['Half']==halfs[0]])),2)
                h2 = round(dP[dP['Half']==halfs[1]]['xG'].sum()/max(1,len(dP[dP['Half']==halfs[1]])),2)
                shootingTable = pd.concat([shootingTable, pd.DataFrame({
                    'PATASTATS INDEX': [f'    {label} '],
                    'Per Match': [per_match],
                    '1st half': [h1],
                    '2nd half': [h2]
                })], ignore_index=True)
            
            # Goals - XG
            dP_goal = playerShot[playerShot['Outcome']=="Goal"]
            dp2 = playerShot
            per_match = len(dP_goal) - round(dp2['xG'].sum(),2)
            h1 = len(dP_goal[dP_goal['Half']==halfs[0]]) - round(dp2[dp2['Half']==halfs[0]]['xG'].sum(),2)
            h2 = len(dP_goal[dP_goal['Half']==halfs[1]]) - round(dp2[dp2['Half']==halfs[1]]['xG'].sum(),2)
            shootingTable = pd.concat([shootingTable, pd.DataFrame({
                'PATASTATS INDEX': ['    Goals - XG '],
                'Per Match': [per_match],
                '1st half': [h1],
                '2nd half': [h2]
            })], ignore_index=True)
            
            # np Goals - np XG
            dP_goal_np = playerShot[(playerShot['Outcome']=="Goal") & (playerShot['Extra 1']!="Penalty")]
            dp2_np = playerShot[playerShot['Extra 1']!="Penalty"]
            per_match = len(dP_goal_np) - round(dp2_np['xG'].sum(),2)
            h1 = len(dP_goal_np[dP_goal_np['Half']==halfs[0]]) - round(dp2_np[dp2_np['Half']==halfs[0]]['xG'].sum(),2)
            h2 = len(dP_goal_np[dP_goal_np['Half']==halfs[1]]) - round(dp2_np[dp2_np['Half']==halfs[1]]['xG'].sum(),2)
            shootingTable = pd.concat([shootingTable, pd.DataFrame({
                'PATASTATS INDEX': ['    np Goals - np XG '],
                'Per Match': [per_match],
                '1st half': [h1],
                '2nd half': [h2]
            })], ignore_index=True)
            
            # Average Shot length
            dP = playerShot
            per_match = round(dP['Actions Pos Length'].mean(),1)
            h1 = round(dP[dP['Half']==halfs[0]]['Actions Pos Length'].mean(),1)
            h2 = round(dP[dP['Half']==halfs[1]]['Actions Pos Length'].mean(),1)
            shootingTable = pd.concat([shootingTable, pd.DataFrame({
                'PATASTATS INDEX': ['    Average Shot Distance (Y)'],
                'Per Match': [per_match],
                '1st half': [h1],
                '2nd half': [h2]
            })], ignore_index=True)
            
            # تحويل الأعمدة إلى نسب مئوية إذا مطلوب
            shootingTable = convert_percent_columns(shootingTable)
            
            return shootingTable
    
    
        shootingTable = create_shooting_table(playerName, dataShot, dataShotGood, halfs)
    
    
    
        def plot_shooting_table(shootingTable, img_path):
            fig_ShotTable, ax_ShotTable = plt.subplots(figsize=(16, 16))
            
            # إضافة صورة خلفية
            img = Image.open(img_path)
            fig_ShotTable.figimage(img, xo=600, yo=450, alpha=0.2, zorder=0)
            
            bg_color = "w"
            text_color = "k"
            plt.rcParams["text.color"] = text_color
            plt.rcParams["font.family"] = "Arial"
            fig_ShotTable.set_facecolor(bg_color)
            ax_ShotTable.set_facecolor(bg_color)
        
            # دالة لتحويل القيم لأرقام آمنة
            def safe_float(x):
                try:
                    val = float(x)
                    if val < 0:
                        return f"- {abs(val):.2f}"
                    else:
                        return f"{val:.2f}"
                except:
                    return str(x)
        
            num_cols = ["Per Match", "1st half", "2nd half"]
            for col in num_cols:
                shootingTable[col] = shootingTable[col].apply(safe_float)
        
            # إنشاء الجدول
            tab = Table(
                shootingTable.round(2),
                cell_kw={"linewidth": 0, "edgecolor": "k","height":1.2},
                textprops={"ha": "right","va":"center","fontsize":12},
                col_label_divider=True,
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
        
            # نصوص العنوان
            fig_ShotTable.text(0.14, 0.9, "Shooting Stats", fontsize=22, color='k')
            fig_ShotTable.text(0.14, 0.892, "____________", fontsize=22, color='gold')
            
            return fig_ShotTable
    
        fig_ShotTable = plot_shooting_table(shootingTable, r"WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg")
    
    
        from mplsoccer import VerticalPitch
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyArrowPatch
        from highlight_text import ax_text
        import numpy as np
        import pandas as pd
    
    
        def plot_shots_map(playerName, dataShot, dataShotGood, img):
            
            # ================================
            # 1. فلترة بيانات اللاعب باستخدام get_player_data
            # ================================
            playerShots = get_player_data(dataShot, playerName)
            playerShotsGood = get_player_data(dataShotGood, playerName)
        
            # ================================
            # تعريف الألوان والماركرات لكل Outcome
            # ================================
            colors = {'Blocked':'gray','Off T':'r','Saved':'gold','Wayward':'none','Post':'gold'}
            edgecolors = {'Blocked':'k','Off T':'none','Saved':'gold','Wayward':'r','Post':'gold'}
            markers = {'Goal':'football','Blocked':'s','Off T':'o','Saved':'o','Wayward':'o','Post':'d'}
            linestyles = {'Goal':'-','Blocked':'-','Off T':'--','Saved':'-','Wayward':'--','Post':'-'}
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
            # 3. رسم التسديدات
            # ================================
            def draw_Shots_scatter(ax, playerShots):
                pitch.draw(ax=ax)
                added_labels = set()
                playerShots = playerShots.dropna(subset=['Actions positions x', 'Actions positions y'])
                for outcome, group in playerShots.groupby('Outcome'):
                    for i, row in group.iterrows():
                        xg_value = row.get('xG', 0)
                        mark = markers.get(outcome, 'o')
                        label_text = labels.get(outcome, '-') if outcome not in added_labels else None
                        linestyle_ = linestyles.get(outcome, '-')
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
                            except:
                                pitch.scatter(
                                    x=row['Actions positions x'],
                                    y=row['Actions positions y'],
                                    s=xg_value*800 if pd.notna(xg_value) else 400,
                                    color='g',
                                    edgecolor='w',
                                    marker='d',
                                    label=label_text,
                                    lw=1,
                                    linestyle=linestyle_,
                                    zorder=4,
                                    ax=ax
                                )
                        else:
                            pitch.scatter(
                                x=row['Actions positions x'],
                                y=row['Actions positions y'],
                                s=xg_value*800,
                                marker=mark,
                                color=color,
                                edgecolor=edgecolor,
                                lw=1,
                                label=label_text,
                                zorder=4,
                                ax=ax
                            )
                        if pd.notna(row.get('Actions positions x End')) and pd.notna(row.get('Actions positions y End')):
                            pitch.arrows(
                                xstart=row['Actions positions x'] + 0.5,
                                ystart=row['Actions positions y'],
                                xend=row['Actions positions x End'],
                                yend=row['Actions positions y End'],
                                width=xg_value*20,
                                color='lime',
                                alpha=min(xg_value*2, 1),
                                ax=ax
                            )
                        if outcome == 'Goal' and row.get('Extra 1') == 'Penalty':
                            pitch.text(
                                row['Actions positions x'],
                                row['Actions positions y'],
                                'P',
                                fontsize=14,
                                fontweight='bold',
                                color='w',
                                ha='center',
                                va='center',
                                zorder=5,
                                ax=ax
                            )
                    added_labels.add(outcome)
        
            draw_Shots_scatter(axs_Shots, playerShots)
        
            # ================================
            # 4. Heatmap أفقي
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
        
            draw_heatmap_full(axs_Shots, playerShots, pitch, statsbomb_cmap_red_blue)
        
            # ================================
            # 5. متوسط المسافة
            # ================================
            if not playerShots['Actions positions x'].dropna().empty:
                avg_ds_shot = round(np.nanmean(playerShots['Actions positions x']))
            else:
                avg_ds_shot = 0
            pitch.lines(120, 75, avg_ds_shot, 75, lw=3, linestyle='--', transparent=True, comet=True, label='Avg Shot Distance', color='gold', ax=axs_Shots)
            axs_Shots.annotate(f'Avg Shot Distance {120-avg_ds_shot}Y', xy=(78, avg_ds_shot+2), zorder=2, ha='center', color='#7c7c7c', weight='bold', fontsize=12, rotation=90)
        
            # ================================
            # 6. نص إحصائي
            # ================================
            total_Shots = len(playerShots)
            Total_xg = round(playerShots['xG'].sum(), 2)
            n_goals = len(playerShots[playerShots['Outcome']=='Goal'])
            n_goals_penalty = len(playerShots[(playerShots['Outcome']=='Goal')&(playerShots['Extra 1']=="Penalty")])
            n_Ontarget = len(playerShotsGood)
            Shot_acc = np.round((n_Ontarget / total_Shots) * 100, 2) if total_Shots > 0 else 0
            ax_title = f'Total Shot: <{total_Shots}> \nGoal : <{n_goals}>    Penalty : <{n_goals_penalty}>\nShot On Target: <{n_Ontarget}> /{total_Shots} |{Shot_acc}% \nTotal xG: <{Total_xg}>'
            ax_text(58, 47, ax_title, ax=axs_Shots, highlight_textprops=[{"color": 'Black'}, {"color":"#56ae6c"},
                                                                         {"color":"#56ae6c"}, {'color':'gold'}, {"color":"#7E1E9C"}], va='center', ha='center', fontsize=20)
        
            # ================================
            # 7. صورة خلفية
            # ================================
            ax_bg = fig_Shots.add_axes([0.75, .86, .15, .15])
            ax_bg.imshow(img, aspect='auto', alpha=1, zorder=-1)
            ax_bg.axis('off')
            pitch.inset_image(80.5, 40, img, height=40, alpha=.2, ax=axs_Shots, zorder=-1)
        
            # ================================
            # 8. سهم اتجاه الهجوم
            # ================================
            arrow = FancyArrowPatch((0.12, 0.26), (0.12, 0.36), arrowstyle='-', linewidth=2, color='k', mutation_scale=20, transform=fig_Shots.transFigure, zorder=2)
            fig_Shots.patches.append(arrow)
            arrow1 = FancyArrowPatch((0.12, 0.56), (0.12, 0.66), arrowstyle='->', linewidth=2, color='k', mutation_scale=20, transform=fig_Shots.transFigure, zorder=2)
            fig_Shots.patches.append(arrow1)
            fig_Shots.text(0.12, 0.41, "Attack Direction", ha='center', va='bottom', fontsize=18, color='k', zorder=3, rotation=90)
        
            # ================================
            # 9. Legend
            # ================================
            marker_handles = []
            for outcome, marker in markers.items():
                if outcome == "Goal":
                    linestyle_ = linestyles.get(outcome, '-')
                    marker_handles.append(axs_Shots.scatter([], [], c='g', linestyle=linestyle_, marker="d", s=300, edgecolor='w'))
                else:
                    linestyle_ = linestyles.get(outcome, '-')
                    color = colors.get(outcome, 'none')
                    edgecolor = edgecolors.get(outcome, 'none')
                    marker_handles.append(axs_Shots.scatter([], [], color=color, linestyle=linestyle_, marker=marker, s=300, edgecolor=edgecolor))
        
            plt.legend(handles=marker_handles, labels=['Goal','Blocked','Off T','Saved','Wayward','Post'], loc='upper center', ncol=1, bbox_to_anchor=(-2.55, -4.62), labelspacing=1.2, fontsize=12, frameon=False, shadow=True)
            fig_Shots.suptitle("Shots Map", fontsize=40, color='gold', y=.965, x=0.3)
        
            return fig_Shots
    
        fig_Shots = plot_shots_map(playerName, dataShot, dataShotGood, img)
    
    
    
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
        
        
        playerShots = get_player_data(dataShot.copy(), selected_player)
        
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
        # النقاط اللي بتحدد العلاقة بين المقياسين
        x_old = np.array([0, 15, 27, 74, 85, 100])
        x_new = np.array([-2, -1.8, 0, 8, 9.8, 10])
    
        # دالة التحويل
        def convert_scale(x):
            return np.interp(x, x_old, x_new)
    
        # تطبيق الدالة على العمود بالكامل
        playerShots['Goal X' ] = playerShots['Goal X' ].apply(convert_scale)
    
    
        #playerShots['Goal X' ]= playerShots['Goal X'] * 1.2 -36
        #y_old = np.array([0, 37, 62, 100])
        #y_new = np.array([5.67, 4, 2.67,  0])
        playerShots['Goal Y' ]=  (100-playerShots['Goal Y'])/15
        
        # ================================
        # تعريف الألوان لكل نوع قدم
        # ================================
        colors = {'Goal':'g','Off T':'r','Saved':'gold','Wayward':'none','Post':'gold'}
        edgecolors = {'Goal':'w','Off T':'none','Saved':'gold','Wayward':'r','Post':'gold'}
        
        # تعريف الـ marker لكل Outcome
        markers = {'Goal':'d','Off T':'o','Saved':'o','Wayward':'o','Post':'d'}
        
        # تعريف linestyle لكل Outcome
        linestyles = {'Goal':'-','Off T':'--','Saved':'-','Wayward':'--','Post':'-'}
        
        # تعريف labels لكل Outcome
        labels = {'Goal':'Goal','Off T':'Off T','Saved':'Saved','Wayward':'Wayward','Post':'Post'}
        
        
        
        
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
                    if outcome == 'Goal' and row.get('Extra 1') == 'Penalty':
                            ax.text(
                                row['Goal X'],
                                row['Goal Y'],
                                'P',
                                fontsize=14,
                                fontweight='bold',
                                color='w',
                                ha='center',
                                va='center',
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
            labels=['Goal','Off T','Saved','Wayward','Post'],
            loc='upper center',
            ncol=5,
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
        ax_Shots_on_frame.text(        x=9.95+.35,
                        y=2.84+.5,
                        s=len(playerShots),
                        weight="bold",
                        fontsize=14,
                        ha="center"
        
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
            (playerShots['Goal Y'] >= 0) & (playerShots['Goal Y'] <= 2.67)
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
                    weight='bold',
                    zorder=5
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
    
    
    
    
        def build_defense_table(playerName):
            DefenseTable = DataFrame()
        
            # ----------------- Defense ACTIONS -----------------
            DefenseTable = addTableRow(DefenseTable, get_player_data(dataChallenge, playerName),
                                       get_player_data(dataChallengeWon, playerName), '    Challenge / Won', 12)
        
            DefenseTable = addTableRow(DefenseTable, get_player_data(dataChallengeAttacking, playerName),
                                       get_player_data(dataChallengeAttackingWon, playerName), '    Att Challenge / Won', 12)
        
            DefenseTable = addTableRow(DefenseTable, get_player_data(dataChallengeDefensive, playerName),
                                       get_player_data(dataChallengeDefensiveWon, playerName), '    Def Challenge / Won', 12)
        
            DefenseTable = addTableRow(DefenseTable, get_player_data(dataTackle, playerName),
                                       get_player_data(dataTackleWon, playerName), '    Tackle / Won', 12)
        
            # ----------------- TACKLES BY ZONE -----------------
            dP = get_player_data(dataTackle, playerName)
            dPG = get_player_data(dataTackleWon, playerName)
        
            # Def 3rd
            mask = dP['Actions positions x'] <= 40
            maskG = dPG['Actions positions x'] <= 40
            DefenseTable = addTableRow(DefenseTable, dP[mask], dPG[maskG], '    Tkl in Def 3rd / won', 12)
        
            # Mid 3rd
            mask = (dP['Actions positions x'] > 40) & (dP['Actions positions x'] <= 80)
            maskG = (dPG['Actions positions x'] > 40) & (dPG['Actions positions x'] <= 80)
            DefenseTable = addTableRow(DefenseTable, dP[mask], dPG[maskG], '    Tkl in Mid 3rd / won', 12)
        
            # Att 3rd
            mask = dP['Actions positions x'] > 80
            maskG = dPG['Actions positions x'] > 80
            DefenseTable = addTableRow(DefenseTable, dP[mask], dPG[maskG], '    Tkl in Att 3rd / won', 12)
        
            # ----------------- AERIAL -----------------
            DefenseTable = addTableRow(DefenseTable, get_player_data(dataAerial, playerName),
                              get_player_data(dataAerialWon, playerName), '    Aerial / Won', 12)
            
            dP = get_player_data(dataAerial, playerName)
            dPG = get_player_data(dataAerialWon, playerName)
        
            # Own Half
            mask = dP['Actions positions x'] > 60
            maskG = dPG['Actions positions x'] > 60
            DefenseTable = addTableRow(DefenseTable, dP[mask], dPG[maskG], '    Aerial in Own Half / won', 12)
        
            # Opp Half
            mask = dP['Actions positions x'] <= 60
            maskG = dPG['Actions positions x'] <= 60
            DefenseTable = addTableRow(DefenseTable, dP[mask], dPG[maskG], '    Aerial in Opp Half / won', 12)
        
            # ----------------- INTERCEPTION -----------------
            dP = get_player_data(interceptionData, playerName)
            dPG = get_player_data(interceptionDataWon, playerName)
        
            # Opp Half
            DefenseTable = addTableRow(DefenseTable, dP, dPG, '    Interception / in Opp Half', 12)
        
            # Own Half
            maskG = dP['Actions positions x'] <= 60
            DefenseTable = addTableRow(DefenseTable, dP, dP[maskG], '    Interception / in Own Half', 12)
        
            # ----------------- TKL + INT -----------------
            tkl = get_player_data(dataTackle, playerName)
            intc = get_player_data(interceptionData, playerName)
            dP = pd.concat([tkl, intc])
            dPG = dP
        
            # Own Half
            maskG = dPG['Actions positions x'] <= 60
            DefenseTable = addTableRow(DefenseTable, dP, dPG[maskG], '     Tkl + Int / in Own Half', 12)
        
            # Opp Half
            maskG = dPG['Actions positions x'] > 60
            DefenseTable = addTableRow(DefenseTable, dP, dPG[maskG], '     Tkl + Int / in Opp Half', 12)
        
            # ----------------- RECOVERED BALLS -----------------
            dP = get_player_data(dataRecoveredBall, playerName)
            dP = dP[dP['Outcome'] != 'Recovery Failure']
        
            # Opp Half
            maskG = dP['Actions positions x'] > 60
            DefenseTable = addTableRow(DefenseTable, dP, dP[maskG], '     Recovered balls / in Opp Half', 12)
        
            # Att 3rd
            maskG = dP['Actions positions x'] > 80
            DefenseTable = addTableRow(DefenseTable, dP, dP[maskG], '     Recovered balls / in Att 3rd', 12)
        
            # ----------------- LOST BALLS -----------------
            dP = get_player_data(dataLostBall, playerName)
            dPG = dP
        
            # Own Half
            maskG = dPG['Actions positions x'] <= 60
            DefenseTable = addTableRow(DefenseTable, dP, dPG[maskG], '     lost balls / in Own Half', 12)
        
            # Def 3rd
            maskG = dPG['Actions positions x'] <= 40
            DefenseTable = addTableRow(DefenseTable, dP, dPG[maskG], '     lost balls / in Def 3rd', 12)
        
            # Own Box
            maskG = (dPG['Actions positions x'] < 18) & (dPG['Actions positions y'] >= 22) & (dPG['Actions positions y'] <= 80-22)
            DefenseTable = addTableRow(DefenseTable, dP, dPG[maskG], '     lost balls / in Own Box', 12)
        
            # ----------------- CLEARANCE & ERROR -----------------
            DefenseTable = addTableRow(DefenseTable, get_player_data(dataClearance, playerName), '-', '     Clearance', 12)
            DefenseTable = addTableRow(DefenseTable, get_player_data(dataError, playerName), '-', '     Error', 12)
        
            # تحويل الأعمدة إلى نسب مئوية لو فيه
            DefenseTable = convert_percent_columns(DefenseTable)
        
            return DefenseTable
    
        DefenseTable = build_defense_table(playerName)
    
    
        def plot_defense_table(DefenseTable,):
            """
            دالة لرسم جدول الدفاع مع صورة خلفية وألوان ودونات.
            
            Parameters:
                DefenseTable (DataFrame): الجدول النهائي للبيانات الدفاعية
                image_path (str): مسار الصورة للخلفية
            """
            
            fig_DefenseTable, ax_DefenseTable = plt.subplots(figsize=(16, 16))
            
            # إضافة الصورة كخلفية
            
            fig_DefenseTable.figimage(img, xo=600, yo=450, alpha=0.2, zorder=0)
            
            # ألوان الصفوف والخلفية
            bg_color = "w"
            text_color = "k"
            
            plt.rcParams["text.color"] = text_color
            plt.rcParams["font.family"] = "Arial"
            
            fig_DefenseTable.set_facecolor(bg_color)
            ax_DefenseTable.set_facecolor(bg_color)
            
            # تحويل القيم الرقمية إلى float آمن مع صيغة 2 أرقام عشرية
            def safe_float(x):
                try:
                    val = float(x)
                    if val < 0:
                        return f"- {abs(val):.2f}"
                    else:
                        return f"{val:.2f}"
                except:
                    return str(x)
            
            num_cols = ["Per Match", "1st half", "2nd half"]
            for col in num_cols:
                if col in DefenseTable.columns:
                    DefenseTable[col] = DefenseTable[col].apply(safe_float)
            
            # رسم الجدول
            tab = Table(
                DefenseTable.round(2),
                cell_kw={"linewidth": 0, "edgecolor": "k", "height": 1.2},
                textprops={"ha": "right", "va": "center", "fontsize": 12},
                col_label_divider=True,
                col_label_divider_kw={"color": "gray", "lw": .45},
                index_col="PATASTATS INDEX",
                even_row_color="w",
                footer_divider=False,
                footer_divider_kw={"color": bg_color, "lw": .5},
                row_divider_kw={"color": "lightgray", "lw": .5},
                column_border_kw={"color": "darkred", "lw": .5},
                column_definitions=[
                    ColumnDefinition("PATASTATS INDEX", title="", textprops={"ha": "left", "fontsize": 14}, width=1.95),
                    ColumnDefinition("Per Match", title="Per 90", textprops={"ha": "center", "fontsize": 12}),
                    ColumnDefinition("1st half"),
                    ColumnDefinition("2nd half"),
                    ColumnDefinition(
                        "Per Match %", title="Per 90 %", width=.7, textprops={"ha": "center"},
                        plot_fn=progress_donut,
                        plot_kw={"is_pct": True, "formatter": "{:.0%}", "radius": 0.49, "color": "r", "width": 0.05, "alpha": .80}
                    ),
                    ColumnDefinition(
                        "1st half %", width=.7, textprops={"ha": "center"},
                        plot_fn=progress_donut,
                        plot_kw={"is_pct": True, "formatter": "{:.0%}", "radius": 0.49, "color": "r", "width": 0.05, "alpha": .80}
                    ),
                    ColumnDefinition(
                        "2nd half %", width=.7, textprops={"ha": "center"},
                        plot_fn=progress_donut,
                        plot_kw={"is_pct": True, "formatter": "{:.0%}", "radius": 0.49, "color": "r", "width": 0.05, "alpha": .80}
                    ),
                ],
            )
            
            # إضافة العنوان
            fig_DefenseTable.text(0.14, 0.9, "\nDefensing Stats", fontsize=22, color='k')
            fig_DefenseTable.text(0.14, 0.892, "_____________", fontsize=25, color='gold')
            
            return fig_DefenseTable
    
        fig_DefenseTable = plot_defense_table(DefenseTable)
    
    
    
        ########################
    
         ## heatmap_defense ##
    
        #######################
    
        # ================================
        # 1. فلترة بيانات اللاعب
        # ================================
    
        playerTackle        = get_player_data(dataTackle, playerName)
        playerTackleWon     = get_player_data(dataTackleWon, playerName)
        playerTacklelost    = playerTackle[~playerTackle.index.isin(playerTackleWon.index)]
        
        playerAerial        = get_player_data(dataAerial, playerName)
        playerAerialWon     = get_player_data(dataAerialWon, playerName)
        playerAeriallost    = playerAerial[~playerAerial.index.isin(playerAerialWon.index)]
        
        playerinterception  = get_player_data(interceptionData, playerName)
        playerRecoveredBall = get_player_data(dataRecoveredBall[dataRecoveredBall['Outcome']!='Recovery Failure'], playerName)
        playerLostBall      = get_player_data(dataLostBall, playerName)
        playerClearance     = get_player_data(dataClearance, playerName)
        playerError         = get_player_data(dataError, playerName)
        playerDefensive     = get_player_data(dataDefensive, playerName)
    
                      
        
        
        halves = ['1st Half', '2nd Half']
        
        def filter_half(data, half):
            return data[data['Half'] == half]
        # ================================
        # 2. إنشاء الشكل والمحاور
        # ================================
        fig_defense, axs_defense = plt.subplots(1, 2, figsize=(16, 9))
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
            Tackle_Won = filter_half(playerTackleWon, half)
            Tackle_lost = filter_half(playerTacklelost, half)
            Aerial_Won = filter_half(playerAerialWon, half)
            Aerial_lost = filter_half(playerAeriallost, half)
            interception = filter_half(playerinterception, half)
            RecoveredBall = filter_half(playerRecoveredBall,half)  
            LostBall = filter_half(playerLostBall,half) 
            Clearance = filter_half(playerClearance,half)
            Error = filter_half(playerError,half)
            
            
            
            # pass_good = pass_good.drop(index=key_pass.index, errors='ignore')
            # pass_bad = pass_bad.drop(index=key_pass.index, errors='ignore')
            
            return Tackle_Won,Tackle_lost,Aerial_Won,Aerial_lost,interception,RecoveredBall,LostBall,Clearance,Error
        
        half_data = [prepare_half_data(h) for h in halves]
        
        # ================================
        # 4. رسم خطوط التمريرات
        # ================================
        def draw_Defensive_actions(ax, Tackle_Won, Tackle_lost, Aerial_Won, Aerial_lost,
                            interception, RecoveredBall, LostBall, Clearance, Error):
            
            pitch.draw(ax=ax)
            pitch.scatter(
                            x=Tackle_Won['Actions positions x'],
                            y=Tackle_Won['Actions positions y'],
                            s=80,
                            c='skyblue',
                            marker="^",
                            edgecolor='skyblue',
                            lw=.75,
                            zorder=4,
                            ax=ax)
        
            pitch.scatter(
                            x=Tackle_lost['Actions positions x'],
                            y=Tackle_lost['Actions positions y'],
                            s=80,
                            c='None',
                            marker="^",
                            edgecolor='skyblue',
                            lw=.75,
                            zorder=4,
                            ax=ax)
        
            pitch.scatter(
                            x=Aerial_Won['Actions positions x'],
                            y=Aerial_Won['Actions positions y'],
                            s=80,
                            c='#7E1E9C',
                            marker="d",
                            edgecolor='#7E1E9C',
                            lw=.75,
                            zorder=4,
                            ax=ax)
            pitch.scatter(
                            x=Aerial_lost['Actions positions x'],
                            y=Aerial_lost['Actions positions y'],
                            s=80,
                            c='None',
                            marker="d",
                            edgecolor='#7E1E9C',
                            lw=.75,
                            zorder=4,
                            ax=ax)
            pitch.scatter(
                            x=interception['Actions positions x'],
                            y=interception['Actions positions y'],
                            s=50,
                            c='gold',
                            marker="P",
                            edgecolor='gold',
                            lw=.25,
                            zorder=4,
                            ax=ax)
            pitch.scatter(
                            x=RecoveredBall['Actions positions x'],
                            y=RecoveredBall['Actions positions y'],
                            s=50,
                            c='#56ae6c',
                            marker="o",
                            edgecolor='#56ae6c',
                            lw=.75,
                            zorder=4,
                            ax=ax)
            pitch.scatter(
                            x=LostBall['Actions positions x'],
                            y=LostBall['Actions positions y'],
                            s=50,
                            c='#ba4f45',
                            marker="o",
                            edgecolor='#ba4f45',
                            lw=.75,
                            zorder=4,
                            ax=ax)
            pitch.scatter(
                            x=Clearance['Actions positions x'],
                            y=Clearance['Actions positions y'],
                            s=80,
                            c='orange',
                            marker="s",
                            edgecolor='orange',
                            lw=.75,
                            zorder=4,
                            ax=ax)
            pitch.scatter(
                            x=Error['Actions positions x'],
                            y=Error['Actions positions y'],
                            s=80,
                            c='#ba4f45',
                            marker="X",
                            edgecolor='#ba4f45',
                            lw=.75,
                            zorder=4,
                            ax=ax)
            
            # ax.axhline(
            #     y=80.5,
            #     xmin=.05, xmax=.95,# موقع الخط على محور X
            #     color='b',    # اللون اللبني السماوي
            #     linestyle='--',     # نوع الخط (ممكن تخليه '-' لو عاوزه متصل)
            #     linewidth=1.5,        # سُمك الخط
            #     zorder=3,           # عشان يكون فوق الخلفية لكن تحت النقاط مثلاً
            #     )
        for i, data in enumerate(half_data):
            draw_Defensive_actions(axs_defense[i], *data)
            axs_defense[i].set_title(f"{halves[i]}", y=1.05, fontsize=22, color='k')
        
        # ================================
        # 5. دالة Heatmap جديدة (أفقي وعمودي)
        # ================================
        def filter_90_percent(df, x_col, y_col):
            # نجيب x و y
            X = df[[x_col, y_col]]
        
            # نحسب IQR لكل بُعد
            Q1 = X.quantile(0.00)
            Q3 = X.quantile(1)
        
            # نسمح فقط بـ 90% من البيانات (من 5% إلى 95%)
            mask = (X >= Q1) & (X <= Q3)
            mask = mask.all(axis=1)
        
            return df[mask]
    
        def draw_heatmap_full(ax, data, pitch, cmap):
            # Heatmap أفقي
            bin_stat_h = pitch.bin_statistic_positional(
                data['Actions positions x'],
                data['Actions positions y'],
                statistic='count',
                positional='horizontal',
                normalize=True
            )
            filtered = filter_90_percent(data, 'Actions positions x', 'Actions positions y')
            hull = pitch.convexhull(filtered['Actions positions x'], filtered['Actions positions y'])
            poly = pitch.polygon(hull, ax=ax, edgecolor='slateblue', facecolor='slateblue', alpha=0.3)
            #hm_h = pitch.heatmap_positional(bin_stat_h, ax=ax, cmap=cmap, edgecolors='None', zorder=2, alpha=.25)
            
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
                data['Actions positions x'],
                data['Actions positions y'],
                statistic='count',
                positional='vertical',
                normalize=True
            )
        
        
            hm_h = pitch.heatmap_positional(bin_stat_v, ax=ax, cmap=cmap, edgecolors='None', zorder=2, alpha=.25)
        
            
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
        
        def safe_combine_dfs(dfs):
            """دمج أكثر من DataFrame مع تجاهل الفارغة وNaN"""
            clean_dfs = [df.dropna(subset=['Actions positions x','Actions positions y']) for df in dfs]
            clean_dfs = [df for df in clean_dfs if not df.empty]
            if clean_dfs:
                return pd.concat(clean_dfs, ignore_index=True)
            else:
                return pd.DataFrame(columns=['Actions positions x','Actions positions y'])
        selected_indices = list(range(9))
        heatmaps = []
        for i, data in enumerate(half_data):
            combined_data = safe_combine_dfs([data[j] for j in selected_indices])
            if not combined_data.empty:
                hm = draw_heatmap_full(axs_defense[i], combined_data, pitch, statsbomb_cmap_red_blue)
                heatmaps.append(hm)
            else:
                heatmaps.append(None)
        # ================================
        # 6. صورة خلفية (اختياري)
        # ================================
        ax_bg = fig_defense.add_axes([0.8, .82, .15, .18])
        ax_bg.imshow(img, aspect='auto', alpha=1)
        ax_bg.axis('off')
        
        # ================================
        # 7. نص أسفل الشكل
        # ================================
        
        colors = {'Tackle Won':'skyblue','Aerial Won':'#7E1E9C','interception':'gold','Clearance':'orange','Recovered Ball':'#56ae6c','Lost Ball':'#ba4f45',"Error":'#ba4f45'}
        
        edgecolors = {'Tackle Won':'skyblue','Aerial Won':'#7E1E9C','interception':'gold','Clearance':'orange','Recovered Ball':'#56ae6c','Lost Ball':'#ba4f45',"Error":'#ba4f45'}
        
        
        # تعريف الـ marker لكل Outcome
        markers = {'Tackle Won':'^','Aerial Won':'d','interception':'P','Clearance':'s','Recovered Ball':'o','Lost Ball':'o',"Error":'X'}
    
        lws = {'interception':.5}
        
        
        
        # تعريف labels لكل Outcome
        labels = {'Tackle Won':'Tackle','Aerial Won':'Aerial','interception':'interception','Clearance':'Clearance','Recovered Ball':'Recovery','Lost Ball':'Lost Ball',"Error":'Error'}
        
        
        
        marker_handles = []
        for outcome, marker in markers.items():
                linestyle_ = linestyles.get(outcome, '-')
                color = colors.get(outcome, 'none')
                edgecolor = edgecolors.get(outcome, 'none')
                lw = lws.get(outcome,1.5)
                marker_handles.append(
                    axs_defense[0].scatter([], [], color=color,  marker=marker, s=200, edgecolor=edgecolor,lw=lw )
                )
        
        
        plt.legend(
            handles=marker_handles,
            labels=['Tackle','Aerial','interception','Clearance','Recovery','Lost Ball','Error'],
            loc='upper center',
            ncol=7,
            bbox_to_anchor=(-2, -5),
            labelspacing=1.2,
            fontsize=12,
            frameon=False,
            shadow=True
        )
        # ================================
        # 8. Colorbar
        # ================================
        cax = fig_defense.add_axes([0.1, 0.15, 0.04, 0.55])
        cbar = fig_defense.colorbar(heatmaps[0][0], cax=cax, location="left", fraction=0.016, pad=0.0,
                                   orientation='vertical', format=StrMethodFormatter("{x:.0%}"))
        cbar.set_label('Defense Actions Intensity', fontsize=20, labelpad=20)
        cbar.ax.tick_params(labelsize=12)
        
        # ================================
        # 9. عنوان الشكل العام
        # ================================
        fig_defense.suptitle("Defense Actions Map", fontsize=40, color='gold', y=.95, x=0.2)
        fig_defense.text(0.5, -0.1, " ", fontsize=40, color='gold')
        # ================================
        # 10. سهم اتجاه الهجوم
        # ================================
        arrow = FancyArrowPatch((0.55, 0.16), (0.55, 0.26), arrowstyle='-', linewidth=2,
                                color='k', mutation_scale=20, transform=fig_defense.transFigure, zorder=2)
        fig_defense.patches.append(arrow)
        
        arrow1 = FancyArrowPatch((0.55, 0.56), (0.55, 0.66), arrowstyle='->', linewidth=2,
                                 color='k', mutation_scale=20, transform=fig_defense.transFigure, zorder=2)
        fig_defense.patches.append(arrow1)
        
        fig_defense.text(0.55, 0.31, "Attack Direction", ha='center', va='bottom',
                        fontsize=18, color='k', zorder=3, rotation=90)
        
        # ================================
        # 11. عرض الشكل
    
    
    
    
    
        def plot_pressing_heatmap(playerName, dataPressure):
            """
            رسم Heatmap لعمليات الضغط باستخدام بيانات اللاعب عبر get_player_data
            """
        
            # -------------------------------
            # 1) فلترة البيانات باستخدام الدالة الموحدة
            # -------------------------------
            PressurPlayer = get_player_data(dataPressure, playerName)
        
            # -------------------------------
            # 2) إنشاء الشكل والملعب
            # -------------------------------
            fig_pressing, ax_pressing = plt.subplots(figsize=(16, 12))
        
            pitch = VerticalPitch(pitch_color='w', line_color='k', line_zorder=2)
            pitch.draw(ax=ax_pressing)
        
            # -------------------------------
            # 3) إنشاء Colormap مخصصة
            # -------------------------------
            base_cmap = plt.colormaps.get_cmap('turbo')
            colors = base_cmap(np.linspace(0, 1, 256))
            colors[:49, :] = np.array([1, 1, 1, 1])  
            custom_cmap = LinearSegmentedColormap.from_list("custom_turbo_white", colors)
        
            # -------------------------------
            # 4) Scatter لنقاط الضغط
            # -------------------------------
            pitch.scatter(
                PressurPlayer['Actions positions x'],
                PressurPlayer['Actions positions y'],
                c='#ff6600',
                edgecolor="k",
                s=80,
                lw=2,
                ax=ax_pressing,
                zorder=2
            )
            pitch.scatter(
                55,
                -7,
                c='#ff6600', edgecolor="gray",marker='^',s=300, lw=1,ax=ax_pressing, label='Good Actions',zorder=5,alpha=.7,
            )
            pitch.scatter(
                60,
                -7,
                c='#ff6600', edgecolor="gray",marker='^',s=300, lw=1,ax=ax_pressing, label='Good Actions',zorder=5,alpha=.7,
            )
            pitch.scatter(
                65,
                -7,
                c='#ff6600', edgecolor="gray",marker='^',s=300, lw=1,ax=ax_pressing, label='Good Actions',zorder=5,alpha=.8,
            )
    
            # -------------------------------
            # 5) تجهيز البيانات للـ KDE
            # -------------------------------
            x = pd.to_numeric(PressurPlayer['Actions positions x'], errors='coerce').dropna()
            y = pd.to_numeric(PressurPlayer['Actions positions y'], errors='coerce').dropna()
           #kde_before = pitch.kdeplot(PressurPlayer['Actions positions x'], PressurPlayer['Actions positions y'], ax=ax_pressing,
           #                   fill=True, levels=200, thresh=0,
           #                   cut=1, cmap=custom_cmap)
    
            # -------------------------------
            # 6) رسم Heatmap إذا البيانات كافية
            # -------------------------------
            if len(x) < 2 or len(y) < 2 or np.std(x) == 0 or np.std(y) == 0:
                print("⚠️ لا يمكن حساب KDE: البيانات قليلة أو ثابتة")
            else:
                # ===============================
                # 🔥 إنشاء الهيت ماب متوافقة مع الاتجاه الرأسي
                # ===============================
                k = gaussian_kde(np.vstack([x, y]))
                xi, yi = np.mgrid[0:120:600j, 0:80:600j]
                
                zi = k(
                    np.vstack([xi.flatten(), yi.flatten()])
                  )
        
                # تدوير الهيت ماب لتتوافق مع الملعب الرأسي
                zi_rot = np.rot90(zi.reshape(xi.shape).T)
        
                ax_pressing.imshow(
                    zi_rot,
                    extent=[0, 80, 0, 120],
                    origin='upper',
                    cmap=custom_cmap,
                    alpha=0.9,
                    zorder=1
                )
            
        
    
            # -------------------------------
            # 7) صور توضيحية
            # -------------------------------
            pitch.inset_image(60, 42, img, height=70, alpha=.42, ax=ax_pressing, zorder=-1)
            pitch.inset_image(135, 130, img, height=35, alpha=1, ax=ax_pressing, zorder=1)
        
            # -------------------------------
            # 8) عنوان
            # -------------------------------
            ax_pressing.set_title(
                "\n\n    Heatmap of Pressing Actions",
                fontsize=35,
                color='gold',
                y=1.02,
                x=-.23
            )
        
            # -------------------------------
            # 9) مؤشر الاتجاه الهجومي
            # -------------------------------
            fig_pressing.text(
                0.32, 0.42, "Attack Direction",
                ha='center', va='bottom',
                fontsize=18, color='gray', rotation=90
            )
        
            # -------------------------------
            # 10) تعديل حدود الملعب
            # -------------------------------
    
            ax_pressing.set_xlim(-15, 85)
            ax_pressing.set_ylim(-10, 130)
    
            return fig_pressing
        fig_pressing = plot_pressing_heatmap(playerName, dataPressure)
    
        def plot_player_report_header(playerName, dataPressure):
            """
            إنشاء صفحة مقدمة تقرير اللاعب أو الفريق
            """
            # إنشاء الشكل
            fig1, ax = plt.subplots(figsize=(16, 9))
            ax.axis('off')
        
            # ===== 1) تجهيز بيانات النص =====
            # اسم اللاعب بدون الجزء الأول لو كان فيه نقطة
            player_name = playerName.split('.', 1)[1] if '.' in playerName else playerName
        
            # بيانات المباراة
            team = dataPressure['Team'].dropna().unique()[0]
            opp  = dataPressure['Opposition'].dropna().unique()[0]
            match_info = f"{team} Vs {opp}"
        
            # تاريخ المباراة
            match_date_raw = dataPressure['Match Date'].dropna().unique()[0]
            match_date = datetime.strptime(match_date_raw, "%m/%d/%Y").strftime("%d %b %Y")
        
            # ===== 2) رسم صورة اللاعب في الثلث الأعلى =====
            fig1.figimage(img, xo=fig.bbox.xmax * 0.3, yo=fig.bbox.ymax * 0.35, alpha=1, zorder=2)
        
            # ===== 3) كتابة النصوص =====
            fig1.text(
                0.5, 0.35,
                "Player Report" if playerName != "Team" else "Team Report",
                ha='center', va='center',
                fontsize=30, fontweight='bold',
                color='black', zorder=3
            )
        
            # إذا playerName ليس فريق، أظهر اسم اللاعب
            if playerName != "Team":
                fig1.text(
                    0.5, 0.25,
                    player_name,
                    ha='center', va='center',
                    fontsize=22, fontweight='bold',
                    color='gold'
                )
        
            # نصوص إضافية
            fig1.text(
                0.5, 0.18,
                match_info,
                ha='center', va='center',
                fontsize=14,
                color='gray'
            )
        
            fig1.text(
                0.5, 0.12,
                match_date,
                ha='center', va='center',
                fontsize=13,
                color='dimgray'
            )
        
            return fig1
        fig1 = plot_player_report_header(playerName, dataPressure)
    
    
    
    
        def plot_all_actions_heatmap(actionData):
            """
            رسم Heatmap للأفعال مع نقاط المتوسط لكل لاعب
            
            Parameters:
            -----------
            actionData : pd.DataFrame
                بيانات الأفعال لجميع اللاعبين
            img : PIL.Image أو array
                صورة توضيحية أو شعار
            
            Returns:
            --------
            fig : matplotlib.figure.Figure
                الشكل النهائي
            """
            
            fig_ActionsPlayer, ax_ActionsPlayer = plt.subplots(figsize=(16, 12))
            pitch = VerticalPitch(pitch_color='w', line_color='k', line_zorder=2)
            pitch.draw(ax=ax_ActionsPlayer)
            
            # Colormap مخصص
            base_cmap = plt.colormaps.get_cmap('turbo')
            colors = base_cmap(np.linspace(0, 1, 256))
            colors[:49, :] = np.array([1, 1, 1, 1])
            custom_cmap = LinearSegmentedColormap.from_list("custom_turbo_white", colors)
            
            # ===== رسم نقطة المتوسط لكل لاعب =====
            player_actions = get_player_data(actionData, playerName)
            players = player_actions['Player 1'].dropna().unique()
            for player in players:
                player_actions_ = player_actions[player_actions['Player 1'] == player]
                if len(player_actions_) >= 5:
                    
                    median_x = player_actions_['Actions positions x'].median()
                    median_y = player_actions_['Actions positions y'].median()
                    player_number = player.split('.', 1)[0] if '.' in player else player
                    
                    pitch.scatter(
                        median_x, median_y,
                        c="k", edgecolor="silver", s=500, lw=2, ax=ax_ActionsPlayer, zorder=2
                    )
                    pitch.text(
                        median_x, median_y, player_number,
                        ha='center', va='center', fontsize=12, color='silver',
                        fontweight='bold', zorder=3, ax=ax_ActionsPlayer
                    )
    
       
            pitch.scatter(
                55,
                -7,
                c='#ff6600', edgecolor="gray",marker='^',s=300, lw=1,ax=ax_ActionsPlayer, label='Good Actions',zorder=5,alpha=.7,
            )
            pitch.scatter(
                60,
                -7,
                c='#ff6600', edgecolor="gray",marker='^',s=300, lw=1,ax=ax_ActionsPlayer, label='Good Actions',zorder=5,alpha=.7,
            )
            pitch.scatter(
                65,
                -7,
                c='#ff6600', edgecolor="gray",marker='^',s=300, lw=1,ax=ax_ActionsPlayer, label='Good Actions',zorder=5,alpha=.8,
            )
              
            # تجهيز البيانات للـ KDE لكل النقاط
            x = pd.to_numeric(player_actions['Actions positions x'], errors='coerce').dropna()
            y = pd.to_numeric(player_actions['Actions positions y'], errors='coerce').dropna()
            
            if len(x) >= 2 and len(y) >= 2 and np.std(x) != 0 and np.std(y) != 0:
                k = gaussian_kde(np.vstack([x, y]))
                xi, yi = np.mgrid[0:120:300j, 0:80:200j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                zi_rot = np.rot90(zi.reshape(xi.shape).T)
                ax_ActionsPlayer.imshow(
                    zi_rot, extent=[0, 80, 0, 120],
                    origin='upper', cmap=custom_cmap, alpha=0.9, zorder=1
                )
            
            # إدراج الصور
            pitch.inset_image(60, 42, img, height=70, alpha=.42, ax=ax_ActionsPlayer, zorder=-1)
            pitch.inset_image(135, 130, img, height=35, alpha=1, ax=ax_ActionsPlayer, zorder=1)
            
            # العنوان ونصوص إضافية
            ax_ActionsPlayer.set_title("Heatmap & Avg Position ", fontsize=35, color='gold', y=1.02, x=-0.23)
            fig_ActionsPlayer.text(0.32, 0.42, "Attack Direction", ha='center', va='bottom', fontsize=18, color='gray', zorder=3, rotation=90)
            
            ax_ActionsPlayer.set_xlim(-15, 85)
            ax_ActionsPlayer.set_ylim(-10, 130)
            
            return fig_ActionsPlayer
        fig_ActionsPlayer =  plot_all_actions_heatmap(actionData)
    
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
    from matplotlib.backends.backend_pdf import PdfPages
    
    # ===== تقسيم الصفحة إلى 3 أعمدة =====
    col1, col2, col3 = st.columns([1, 2, 1])  # العمود الأوسط أوسع قليلاً
    
    # ===== زر عرض التقرير =====
    if st.button("عرض التقرير", key="show_report"):
        try:
            # نضع الكونتينر داخل العمود الأوسط فقط
            with col2:
                
    
                # ===== غلاف التقرير =====
                img = Image.open("WhatsApp Image 2025-09-04 at 01.18.28_529ef486.jpg")
                fig_cover, ax_cover = plt.subplots(figsize=(16, 9))
                ax_cover.imshow(img)
                ax_cover.axis('off')
                st.pyplot(fig_cover)
                st.title(f" {selected_player} 📑")
                st.markdown("---")
    
                # ===== التبويبات =====
                tab1, tab2, tab3, tab4 = st.tabs([" ⚽ تمريرات ", "🎯 تسديدات", "📊 إحصائيات","🛡️ تدخلات دفاعيه "])
    
                with tab1:
                    st.subheader("⚽ التمريرات")
                    try:
                        st.pyplot(fig_PassTable)
                        st.pyplot(fig_passes)
                    except:
                        st.info("🚫 لا توجد بيانات للتمريرات.")
    
                with tab2:
                    st.subheader("🎯 التسديدات ")
                    try:
                        st.pyplot(fig_ShotTable)
                        st.pyplot(fig_Shots)
                        st.pyplot(fig_Shots_on_frame)
                    except:
                        st.info("🚫 لا توجد بيانات للتسديدات.")
    
                with tab3:
                    st.subheader("📊 إحصائيات عامة")
                    try:
                        st.pyplot(fig)
                        st.pyplot(fig_ActionsPlayer)
                        st.pyplot(fig_pressing)
                    except:
                        st.info("🚫 لا توجد إحصائيات عامة.")
                with tab4:
                    st.subheader("🛡️ التدخلات الدفاعيه")
                    try:
                        st.pyplot(fig_DefenseTable)
                        st.pyplot(fig_defense)
                    except:
                        st.info("🚫  لا توجد بيانات التدخلات الدفاعيه.")
    
            
            
            st.success("✅ تم عرض التقرير بنجاح.")
    
        except Exception as e:
            st.error(f"حدث خطأ أثناء عرض التقرير: {e}")
    
    from io import BytesIO
    from matplotlib.backends.backend_pdf import PdfPages
    
    # ===================== زر تحميل PDF مباشر =====================
    def generate_and_download_pdf(selected_player):
        # إنشاء PDF في الذاكرة
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig)
            pdf.savefig(fig_ActionsPlayer, bbox_inches='tight', pad_inches=.4)
            pdf.savefig(fig_pressing, bbox_inches='tight')
            pdf.savefig(fig_PassTable)
            pdf.savefig(fig_passes)
            pdf.savefig(fig_ShotTable)
            pdf.savefig(fig_Shots_on_frame, bbox_inches='tight', pad_inches=.25)
            pdf.savefig(fig_DefenseTable)
            pdf.savefig(fig_defense, bbox_inches='tight', pad_inches=0.75)
        plt.close('all')
    
        # العودة إلى بداية البايت ستريم
        pdf_buffer.seek(0)
        return pdf_buffer
    
    
    
    # ========== زر التحميل ==========
    if st.button("⬇️ توليد و تحميل PDF"):
        pdf_data = generate_and_download_pdf(selected_player)
        st.download_button(
            label="تحميل التقرير PDF",
            data=pdf_data,
            file_name=f"PATA_STATS_{selected_player}_Report.pdf",
            mime="application/pdf"
        )

    if st.sidebar.button("🔒 تسجيل الخروج"):
        st.session_state["logged_in"] = False
        st.session_state.pop("username", None)
        st.rerun()


# ----------------- نقطة البداية -----------------
if "page" not in st.session_state:
    st.session_state["page"] = "login"
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "admin" in st.query_params:
    admin_panel()
elif st.session_state["logged_in"]:
    main_menu()
else:
    if st.session_state["page"] == "login":
        login()
    elif st.session_state["page"] == "signup":
        signup()
