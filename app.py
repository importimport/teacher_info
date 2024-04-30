import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import SparkApi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jieba
from wordcloud import wordcloud
import os
import re
import zipfile
from PIL import Image
from io import BytesIO
import joblib
import plotly.graph_objs as go
import plotly.express as px
from pyvis.network import Network

import json
from streamlit_echarts import Map
from streamlit_echarts import JsCode
import streamlit as st
from streamlit_echarts import st_echarts
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts


def get_dataset():
    uploaded_file = st.file_uploader("请上传一个表格", type=["xls", "xlsx", "csv"])
    if uploaded_file is not None:
        head, sep, tail = str(uploaded_file.name).partition(".")
        st.write("文件名称是：" + str(head))
        st.write("文件类型是：" + str(tail))
        if tail == "xls" or tail == "xlsx":
            df = pd.read_excel(uploaded_file)
            column = df.columns
            st.table(df.head(3))
            return df
        elif tail == "csv":
            df = pd.read_csv(uploaded_file,encoding='utf8')
            st.table(df.head(3))
            return df


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text


def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length


def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text

text = []

def app():
    filtered_df = pd.DataFrame()
    st.set_page_config(page_title="爬虫与本地知识库系统",  layout="wide")
    sysmenu = '''
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    '''
    st.markdown(sysmenu,unsafe_allow_html=True)
    with st.sidebar:
        choose = option_menu("智能系统", ["自然基金检索","爬虫", "本地知识库大模型问答"],
                             icons=['translate','house', 'file-earmark-music'],
                             menu_icon="broadcast", default_index=0)
    if choose == "自然基金检索":
        df=get_dataset()
        if  df is not None:
            chromedriver_path = './chromedriver.exe'
            driver = webdriver.Chrome(chromedriver_path)
            wait = WebDriverWait(driver, 10)
            # 打开目标网页
            driver.get("https://www.letpub.com.cn/index.php?page=login")
            time.sleep(2)

            name_input = driver.find_element(By.ID, "email")
            name_input.send_keys("2538134102@qq.com")

            passsword_input = driver.find_element(By.ID, "password")
            passsword_input.send_keys("123456789Jiang")

            denglu_button = driver.find_element(By.XPATH, "//img[@onclick='login();']")
            denglu_button.click()

            # 再次等待，以便查询结果加载
            time.sleep(5)  # 等待5秒，等待查询结果
            driver.get("https://www.letpub.com.cn/index.php?page=grant")
            time.sleep(2)  # 等待2秒，确保页面完全加载
            driver.refresh()
            time.sleep(2)
            # 遍历 DataFrame 中的每一行
            for index, row in df.iterrows():
                # 定位姓名输入框并输入姓名
                person_input = driver.find_element(By.ID, "person")
                person_input.send_keys(row['姓名'])

                # 定位学校输入框并输入学校名称
                company_input = driver.find_element(By.ID, "company")
                company_input.send_keys(row['学校'])

                # 定位到下拉菜单
                select_element = driver.find_element(By.ID, "startTime")
                # 用 Select 类选择下拉菜单
                select_object = Select(select_element)
                # 选择 value 为 "2022" 的选项
                select_object.select_by_value("2022")

                # 定位查询按钮并点击
                submit_button = driver.find_element(By.ID, "submit")
                submit_button.click()
                time.sleep(1)
                try:
                    # Locate the button and click it
                    button = driver.find_element(By.ID, "download_excel_button")
                    button.click()
                    st.success(row['姓名']+" 数据已下载")
                except:
                    st.write(row['姓名']+" 没有找到数据")  # 如果在指定时间内没找到按钮，则输出信息
                time.sleep(1)
                driver.refresh()
            driver.quit()


    elif choose == "爬虫":

        # 主页面设置
        st.title("教师信息查询")


        # 加载数据
        data=load_data("./data/详情页结果.csv")
        data = data.drop_duplicates(subset=['学校', '姓名'])
        # 创建搜索框，允许用户输入搜索关键词
        search_query = st.text_input("请输入学校名或教师名进行搜索：")
        st.session_state['filtered_data'] =[]
        # 如果有输入，则处理输入并过滤数据
        if search_query:
            # 处理输入，支持空格或逗号作为分隔符
            search_terms = [term.strip() for term in re.split(r'[ ,]+', search_query)]
            # 根据分割的关键词长度决定搜索逻辑
            if len(search_terms) == 1:
                # 如果只有一个搜索词，则在两列中搜索
                filtered_data = data[data['学校'].str.contains(search_terms[0], case=False, na=False) |
                                     data['姓名'].str.contains(search_terms[0], case=False, na=False)]
                st.session_state['filtered_data'] = filtered_data['姓名']  # 保存到会话状态中
            elif len(search_terms) >= 2:
                # 如果有两个或更多搜索词，则假设第一个为学校名，第二个为教师名
                filtered_data = data[data['学校'].str.contains(search_terms[0], case=False, na=False) &
                                     data['姓名'].str.contains(search_terms[1], case=False, na=False)]
                st.session_state['filtered_data'] = str(filtered_data['姓名'].tolist()[0])  # 保存到会话状态中
            else:
                filtered_data = data
                st.session_state['filtered_data'] = str(filtered_data['姓名'].tolist()[0]) # 保存到会话状态中
        else:
            filtered_data = data

        # 创建超链接列
        filtered_data['信息'] = filtered_data['信息'].apply(lambda x: f'<a href="{x}">{x}</a>')

        # 将过滤后的DataFrame显示为HTML表格
        # st.write(filtered_data.head(500).to_html(escape=False), unsafe_allow_html=True)
        # Create a Styler object and set the CSS property 'white-space' to 'nowrap' for the cells
        styled_data = filtered_data[['学校','姓名','信息']].head(500).style.set_properties(**{'white-space': 'nowrap'})
        # Convert the Styler to HTML and then display it using Streamlit
        st.write(styled_data.to_html(escape=False), unsafe_allow_html=True)


    elif choose == "本地知识库大模型问答":
        data = load_data("./data/详情页结果.csv")
        data = data.drop_duplicates(subset=['学校', '姓名'])
        # 以下密钥信息从控制台获取
        appid = "8d797093"
        api_secret = "MDRmNTFhZTE0Nzk2Y2VmNzQ0Zjk5MDZm"
        api_key = "a4477d6a14b1d1b8b4ee0102e42810b7"

        # 用于配置大模型版本
        domain = "generalv3"

        # 云端环境的服务地址
        Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"  # v3.0环境的地址


        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        st.success("欢迎与星火大模型进行交流")
        user_input = st.chat_input("请输入您的问题，按回车键提交！")

        if user_input and not st.session_state['filtered_data'].empty:
            x = st.session_state['filtered_data'].tolist()[0]
            print('**********', str(data[data['姓名'] == x]['相关文本'].values))
            # 假设用户输入的是相关的关键词，我们可以使用这些关键词在 filtered_data 中搜索
            user_input2 = user_input+str(data[data['姓名'] == x]['相关文本'].values)
            progress_bar = st.empty()
            with st.spinner("内容已提交，星火大模型正在作答中！"):
                question = checklen(getText("user", user_input2))
                SparkApi.answer = ""
                SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
                feedback = getText("assistant", SparkApi.answer)[1]["content"]
                if feedback:
                    progress_bar.progress(100)
                    st.session_state['chat_history'].append((user_input, feedback))
                    for i in range(len(st.session_state["chat_history"])):
                        user_info = st.chat_message("user")
                        user_content = st.session_state["chat_history"][i][0]
                        user_info.write(user_content)

                        assistant_info = st.chat_message("assistant")
                        assistant_content = st.session_state["chat_history"][i][1]
                        assistant_info.write(assistant_content)

                    with st.sidebar:
                        if st.sidebar.button("清除对话历史"):
                            st.session_state["chat_history"] = []

                else:
                    st.info("对不起，我回答不了这个问题，请你更换一个问题，谢谢！")







    elif choose == "故障分类":

        lunkuo = []
        for root, dirs, files in os.walk(r"E:\notebopk\mystreamlit\graph"):
            for file in files:
                filename = os.path.join(root, file)
                lunkuo.append(filename)

        ziti = []
        for root, dirs, files in os.walk(r"E:\notebopk\mystreamlit\font"):
            for file in files:
                filename = os.path.join(root, file)
                ziti.append(filename)
        uploadedfile = st.file_uploader("请上传词云内容txt", type=["txt"])
        if uploadedfile is not None:
            text = pd.read_table(uploadedfile, encoding='utf-8')
            cut_text = jieba.cut(str(text).replace("columns", "").replace("row", ""))
            wc = wordcloud.WordCloud(
                font_path=st.selectbox("请选择一种字体", (ziti)),  # 字体路劲
                background_color='white',  # 背景颜色
                width=1000,
                height=1000,
                max_font_size=80,  # 字体大小
                min_font_size=1,
                mask=np.array(Image.open(st.selectbox("请选择一种轮廓图", (lunkuo)))),
                max_words=10000
            )
            wc.generate(" ".join(cut_text))
            wc.to_file('ciyun.png')
            col1, col2, col3 = st.columns([0.2, 1, 0.2])
            with col1:
                st.empty()
            with col2:
                st.image('ciyun.png', use_column_width='auto', caption='生成的词云图', output_format="png")
            with col3:
                st.empty()

        else:
            st.warning("你需要上传词云内容文本文件")


    elif choose == "载荷预测":
        selecte2 = option_menu(None, ["Train", "Predict"],
                               icons=['house', 'cloud-upload'],
                               menu_icon="cast", default_index=0, orientation="horizontal")
        if selecte2 == "Train":

            # col1, col2,col3 = st.columns(3)
            df=get_dataset()
            if df is not None:
                X=df['微博内容']
                y=df['标签']
                vector_name=st.selectbox('请选择词向量', ('CountVector', 'Tfidf', 'word2vec'))
                X_vector = get_vector(X, vector_name)
                st.write(X_vector.shape)

                classifier_name = st.selectbox('请选择分类模型', ('KNN', 'SVM', 'DecisionTree'))
                params = add_parameters_ui(classifier_name)
                print(params)
                clf = get_clf(classifier_name, params)
                X_train, X_test, y_train, y_test = train_test_split(X_vector, y, random_state=42, test_size=0.2)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write('Model Accuracy: ',round(acc,3))
                joblib.dump(clf, 'model.pkl')

        elif selecte2 == "Predict":

            st.title("文本分类预测")
            vector_name = st.selectbox('请选择词向量', ('CountVector', 'Tfidf', 'word2vec'))
            # age = st.number_input("文本标题")
            text = pd.Series(st.text_area("输入文本", "请输入文本内容..."))
            # model = joblib.load("model.pkl")
            if st.button("预测"):
                prediction = model.predict(text)
                if prediction[0] == 0:
                    st.write("预测结果：无心脏病")
                else:
                    st.write("预测结果：有心脏病")





    elif choose == "故障诊断机器人":
        selecte5 = option_menu(None, ["Javascript", "展示PPT", "嵌入PDF"],
                               icons=['house', 'cloud-upload', "list-task"],
                               menu_icon="cast", default_index=0, orientation="horizontal")

        if selecte5 == "Javascript":
            html.iframe("https://mp.weixin.qq.com/s/Sr4_IAK3pGWRLgjO51i8Mw")

        elif selecte5 == "展示PPT":
            html.iframe("https://mp.weixin.qq.com/s/i0VcKUHBCEHjoOYvoiGolQ")


        elif selecte5 == "嵌入PDF":
            html.iframe("https://mp.weixin.qq.com/s/W8DX74LZYdosDUXUIpoa1g")
if __name__ == '__main__':
    app()