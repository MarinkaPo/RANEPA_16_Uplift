import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import os
from sklift.metrics import uplift_at_k, uplift_by_percentile, qini_auc_score, qini_curve, uplift_curve
from sklift.viz import plot_qini_curve, plot_uplift_curve
from sklift.models import SoloModel, TwoModels, ClassTransformation

import catboost

import tools
ADD_IMG_PATH = 'additional files'

with st.sidebar:
    st.markdown(''' # Содержание:''')

    st.markdown("## [1. Актуальность тематики](#about)", unsafe_allow_html=True) # https://i.imgur.com/iIOA6kU.png
    st.markdown("## [2. Задача](#task)", unsafe_allow_html=True)
    st.markdown("## [3. Этапы разработки кейса](#pipeline)", unsafe_allow_html=True)
    st.markdown("## [4. Информация о датасете](#data)", unsafe_allow_html=True)
    st.markdown("## [5. Блок 1: Анализ выборки](#analyze)", unsafe_allow_html=True)
    st.markdown("## [6. Блок 2: Самостоятельный выбор клиентов для отправки рекламы](#student_choise)", unsafe_allow_html=True)
    st.markdown("## [7. Блок 3: Подбор стратегии с помощью ML](#ml_models)", unsafe_allow_html=True)
    st.markdown("## [8. Блок 4: Подбор стратегии, исходя из бюджета](#sreategy_budget)", unsafe_allow_html=True)

#-----------------------------------------------------#


# print(os.listdir())
# # os.chdir('../')
# print(os.listdir())
# ---------------------Header---------------------
st.markdown('''<h1 style='text-align: center; color: #000000;'
            >Выбор рациональной маркетинговой стратегии с использованием uplift-моделирования</h1>''', 
            unsafe_allow_html=True)
# st.markdown('''<h3 style='text-align: center; color: #f98e4a;'
#             >(uplift modeling laboratory work)</h3>''', 
#             unsafe_allow_html=True)
			
add_img_file = os.path.join(ADD_IMG_PATH, 'uplift_title_pic.jpg')
title_image = Image.open(add_img_file)
st.image(title_image, use_column_width='auto') # use_column_width='auto') #width=450

# col1, col2, col3 = st.columns([1,12,1])
# with col1:
# 	st.write("")
# with col2:
# 	st.image("additional files/uplift_header_pic.jpg")  # uplift_header_pic.jpg Uplift_modeling.jpg
# with col3:
# 	st.write("")

# with open("src/style.css") as f:
#     st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
# 	title_image = Image.open('additional files/Uplift_modeling.jpg')
# 	st.image(title_image, width=550) # use_column_width='auto') #width=450

st.write("""
Лабораторная работа **"Uplift-моделирование"** знакомит студентов с основами поиска оптимальной стратегии коммуникаций бизнеса со своими клиентами. 
\nЦель uplift-моделирования - оценить чистый эффект от коммуникации, выбрать тех клиентов, которые совершат целевое действие только при специальном 
взаимодействии с ними, а также рассчитать рациональность такого взаимодействия. 

\nЛаборатрная работа состоит из **4х блоков**: 
\n* **Анализ данных:** анализ выборки с помощью методов визуализации;
\n* **Ручной выбор клиентов для отправки рекламы:** вам будет предложено самостоятельно выбрать те группы клиентов, которым будет отправлена реклама. В конце этого блока надо будет найти такую выборку клиентов, чтобы рекламная кампания была прибыльной;
\n* **Выбор клиентов для отправки рекламы с помощью моделей машинного обучения:** вы посмотрите, как с этой же задачей справляются модели машинного обучения и сравните со своим решением.
\n* **Подбор стратегии, исходя из бюджета и цены на рекламу:** найдя точку безубыточности, вы выберете наиболее рациональный путь расходования бюджета.
\n*Ссылка на источник данных:* [_The MineThatData E-Mail Analytics And Data Mining Challenge_](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)""")

st.header('Актуальность тематики', anchor='about') 
st.write(""" \n##### **Кому будет полезна эта лабораторная работа и почему?**
\n* **Студентам управленческих специальностей:**
\nВо время выполнения лабораторной работы вы "примерите" на себя роль руководителя маркетингового направления, произведёте анализ имеющихся данных о клиентах, 
оцените возможность применения новой стратегии, исходя из имеющегося бюджета.
\n* **Студентам маркетинговых направлений:**
\nИз теоритического блока вы узнаете об uplift-моделировании, как одном из популярных методов выбора клиентов для коммуникации. Попробуете применять разные 
модели для uplift-моделирования, а также оцените рентабельность рекламной кампании, в зависимости от стоимости коммуникации с клиентом.
\n* **Студентам направления аналитики и консалтинга:**
\nПервые два блока лабораторной работы посвящены самостоятельному анализу и отбору клиентов для рекламной коммуникации. Используя графический анализ данных 
и сводные таблицы вы сделаете заключение о ваших клиентах и их реакции на прошлую рекламную кампанию.
\n* **Студентам финансовых и экономических специальностей:**
\nДанная лабораторная работа даёт базовые знания о возможностях коммуникаций бизнеса с клиентами. Общее понимание данной темы помогает контроллировать распределение бюджета компании, включая маркетинговые расходы.
\n* **Студентам других специальностей:**
\nДля общего понимания тематики маркетинговых исследований.
""")


st.header('Задача:', anchor='task') 
st.write(""" \nПредставьте, что вы - руководитель маркетингового направления в крупной российской компании. Вы работаете с широкой аудиторией, 
предлагаете покупателям скидки, запускает акции и предложения.

Ваше руководство озабочено появлением нового конурента и ставит вам задачу разработать стратегию воздействия на клиентов через коммуникацию с ними.
Естественно, ваша стратегия должна быть направлена не только на то, чтобы напомнить клиентам о себе, но и на получение прибыли от такой кампании.
\nТаким образом, вам необходимо:
\n**разработать такую стратегию воздействия на клинентов, чтобы при минимальных затратах она принесла максимум прибыли компании**

\nДанные подготовили сотрудники ЛИА РАНХиГС.
""")
#-------------------------Pipeline & Info description-------------------------
st.header('Этапы разработки кейса', anchor='pipeline')
add_img_file = os.path.join(ADD_IMG_PATH, 'Uplift_pipeline.png')
img_pipeline = Image.open(add_img_file) 
st.image(img_pipeline, use_column_width='auto', caption='Схема (пайплайн) лабораторной работы') #width=450

pipeline_bar = st.expander("Описание пайплайна лабораторной работы")
pipeline_bar.markdown(
    """
    \n**Этапы:**
    \n*(зелёным обозначены этапы, работа с которыми доступна студенту, красным - этапы, доступные для корректировки сотрудникам ЛИА)*
    \n**1. Сбор данных:**
    \nБыл использован датасет от консалтинговой компании MineThatData, которая предоставляет консультационные услуги бизнесу для лучшего понимания взаимодействия между клиентами, рекламой, продуктами, брендами и каналами сбыта [(ссылка на данные)](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html);
    \n**2. Анализ данных и поиск закономерностей:**
	\nС помощью графического анализа данных студент исследует клиентов розничного магазина по различным срезам.
    \nИнструменты для этого: библиотеки [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [plotly](https://plotly.com/python-api-reference/index.html);
    \n**3. Ручной выбор клиентов для отправки рекламы:**
	\nВ этом блоке студент:
	\nа) вручную выберет группу клиентов для отправки новой рекламы;
	\nб) оценит рентабельность своего выбора;
	\nв) проанализирует метрики качества по своему решению.
	\nС использованием библиотек  [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [scikit-uplift](https://www.uplift-modeling.com/en/latest/api/index.html), [plotly](https://plotly.com/python-api-reference/index.html);
	\n**4. Работа с моделями машинного обучения для uplift-моделирования этой же задачи:**
	\nСюда входит:
	\na)-б) Выбор моделей машинного обучения, подбор лучших гиперпараметов, обучение и валидация моделей с ними:
	\nБыли использованы модели [CatBoost](https://catboost.ai/en/docs/), [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) и [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
    \nв) Оценка метрик качества моделей
	\nТакже были использованы библиотеки [scikit-learn](https://scikit-learn.org/stable/modules/classes.html), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [plotly](https://plotly.com/python-api-reference/index.html);
    \n**5. Сравнение результатов:**
	\nСтудент сравнивает метрики качества собственного выбора клиентов для отправки рекламы и решения этой же задачи с помощью моделей машинного обучения;
    \n**6. Оформление микросервиса Streamlit, выгрузка на сервер:**
	\nПроводится сотрудником лаборатории, используется студентами РАНХиГС
    \nС использованием библиотеки [streamlit](https://docs.streamlit.io/library/get-started)
    """)
info_bar = st.expander("Информация о целях и применении uplift-моделирования для бизнес-задач")
info_bar.markdown('''
\nВ современном мире коммуникации с клиентами - неотъемлимая часть любого бизнеса. Большинство компаний заинтересовано в предложении своих услуг как можно большему числу клиентов и пользователей их сервисов. 
И этим компаниям важно знать как оценивать эффект от коммуникации со своими пользователями. Помочь оценить данный эффект и выбрать интересующую группу пользователей могут модели Uplift-моделирования.
\nЦелевым результатом применения Uplift модели является определение клиентов, взаимодействие с которыми принесет компании наиболее интересуемый эффект. Всех адресатов различных рассылок можно разделить на 4 группы:''')

col1, col2, col3 = info_bar.columns([1,3,1])
with col1:
	st.write("")
with col2:
	add_img_file = os.path.join(ADD_IMG_PATH, 'Uplift_modeling.jpg')
	st.image(add_img_file)  # uplift_header_pic.jpg Uplift_modeling.jpg
with col3:
	st.write("")

# uplift_image = Image.open('additional files/Uplift_modeling.jpg')
# info_bar.image(uplift_image, width=450) # use_column_width='auto') #width=450

info_bar.markdown('''\n**1.** Клиент, который **отреагирует негативно**, если отправить ему предложение (отпишется от рассылок, откажется от услуг сервиса и т.д.)
\n**2.** "Потерянный" клиент, который **никогда не совершит действие**, неважно отправить ему предложение или нет.
\n**3.** "Лояльный" клиент, который **в любом случае совершит действие**, даже если не отправлять ему предложение.
\n**4.** "Убеждаемый" клиент, который **совершит действие, если отправить ему предложение**.

Первые три типа не принесут дополнительного дохода от рекламной кампании, но могут принести убытки из-за особенности реакции пользователей или затраты на способы взаимодействия с ними. 
Из всех пользователей, нас больше всего интересуют пользователи *четвертого типа*, которые ответят на предложение если напрямую отправить им его. Найти данную категорию пользователей можно при помощи Uplift-моделирования. 
\nUplift модель оценивает разницу в поведении адресата при наличии воздействия (или предложения) и при его отсутствии.
\nНо перед этим очень важно собрать данные, на которых можно «обучиться». Для этого собирают обучающий набор – абсолютно похожие друг на друга покупатели. Например, в группу объединяется 1000 человек со схожим поведением в плане предпочтения продуктов. Среди них выбирают 500 случайных людей, отправляют им рассылку о скидке, остальные остаются без предложения.

Далее важно проследить за изменениями в какой-либо метрике – среднем чеке или конверсии. Изменения в этом случае происходят только в результате коммуникации, например рассылки.

Например, если клиенты покупали товар в 70% случаев, а со скидкой купили в 80% ситуаций, значит, Uplift составляет 10%. В результате можно решить, выгодна ли скидка или коммуникация с клиентом.
На основе этих данных скидки и предложения рассылаются целенаправленно.

Кроме того, выполнение целевого действия сильно зависит от различных характеристик самой кампании, например, канала взаимодействия или типа и размера предлагаемого маркетингового предложения. 
Для максимизации прибыли следует также подбирать и эти параметры.
''')

#-----------------------------БЛОК Информации-----------------------------#
st.header('Информация о датасете', anchor='data')	
st.markdown("""Наш набор данных содержит 42 693 строки с данными клиентов, которые в последний раз совершали покупки в течение двенадцати месяцев.

Из данных уже отделена тестовая выборка в виде 30% записей клиентов, так что данных в предоставленной выборке будет меньше.

Среди клиентов была проведена рекламная кампания с помощью _email_ рассылки:
- Половине клиентов были выбраны случайным образом для получения электронного письма, рекламирующего женскую продукцию;
- С оставшейся половиной коммуникацию не проводили.

Для каждого клиента из выборки замерили факт перехода по ссылке в письме, факт совершения покупки и сумму трат за
две недели, следующими после получения письма.

Пример данных приведен ниже. """)


# загрузим датасет
dataset, target, treatment = tools.get_data()

# загрузим предикты моделей
ct_cbc = pd.read_csv('src/model_predictions/catboost/ct_cbc.csv', index_col='Unnamed: 0')
sm_cbc = pd.read_csv('src/model_predictions/catboost/sm_cbc.csv', index_col='Unnamed: 0')
tm_dependend_cbc = pd.read_csv('src/model_predictions/catboost/tm_dependend_cbc.csv', index_col='Unnamed: 0')
tm_independend_cbc = pd.read_csv('src/model_predictions/catboost/tm_independend_cbc.csv', index_col='Unnamed: 0')

tm_rfc = pd.read_csv('src/model_predictions/random_forest/tm_rfc.csv', index_col='Unnamed: 0')

sm_xgboost = pd.read_csv('src/model_predictions/xgboost/sm_xgb.csv', index_col='Unnamed: 0')

# загрузим данные
data_train_index = pd.read_csv('data/data_train_index.csv')
data_test_index = pd.read_csv('data/data_test_index.csv')
treatment_train_index = pd.read_csv('data/treatment_train_index.csv')
treatment_test_index = pd.read_csv('data/treatment_test_index.csv')
target_train_index = pd.read_csv('data/target_train_index.csv')
target_test_index = pd.read_csv('data/target_test_index.csv')

# фиксируем выборки, чтобы результат работы ML был предсказуем
data_train = dataset.loc[data_train_index['0']]
data_test = dataset.loc[data_test_index['0']]
treatment_train = treatment.loc[treatment_train_index['0']]
treatment_test = treatment.loc[treatment_test_index['0']]
target_train = target.loc[target_train_index['0']]
target_test = target.loc[target_test_index['0']]


refresh = st.button('Обновить и показать выборку', key='button1')
title_subsample = data_train.sample(7)
if refresh:
	title_subsample = data_train.sample(7)

fig = go.Figure(data=[go.Table(
                                columnorder = [n for n in range(1, len(title_subsample.columns)+1)],
                                columnwidth = [0.9,1.8,1,1,1.1,1.5,1,1.3,0.9], # *len(title_subsample.columns),
                                header=dict(values=list(title_subsample.columns),
                                            line_color='black',
                                            fill_color='#d9d9d9',
                                            align=['center']*len(title_subsample.columns),
                                            font=dict(color='black', size=14),
                                            height=30),                                            
                                cells=dict(values=[title_subsample[col] for col in title_subsample.columns],
                                            fill_color='white',
                                            align=['center']*len(title_subsample.columns),
                                            font=dict(color='black', size=14),
                                            height=30))
                              ])
fig.update_traces(cells_line_color='black')
fig.update_layout(width=700, height=245, margin=dict(b=0, l=0, r=1, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
st.plotly_chart(fig)
# st.dataframe(title_subsample) #, width=500)
st.write(f"Всего записей: {data_train.shape[0]}")

st.write('*Описание данных:*')
st.markdown(
	"""
		| Колонка           | Обозначение                                                            |
		|-------------------|------------------------------------------------------------------------|
		| _recency_         | Число месяцев с момента последней покупки                              |
		| _history_segment_ | Классификация клиентов в долларах, потраченных в прошлом году          |
		| _history_         | Фактическая стоимость в долларах, потраченная в прошлом году           |
		| _mens_            | Флаг 1/0, 1 = клиент приобрел мужские товары в прошлом году            |
		| _womens_          | Флаг 1/0, 1 = клиент приобрел женские товары в прошлом году            |
		| _zip_code_        | Классифицирует почтовый индекс как городской, пригородный или сельский |
		| _newbie_          | Флаг 1/0, 1 = Новый клиент за последние двенадцать месяцев             |
		| _channel_         | Описывает каналы, через которые клиент приобрел тоовар в прошлом году  |

		---
	"""
	)

#-----------------------------БЛОК 1 АНАЛИЗ ДАННЫХ-----------------------------#
st.header('Блок 1: Анализ выборки', anchor='analyze')
st.markdown('''#### Задание по Блоку 1:
\nДля того, чтобы лучше понять на какую аудиторию следует запустить новую рекламную кампанию, проведите небольшой анализ данных, используя графики ниже и отвечая на вопросы по ним.''')
# with st.expander('Развернуть блок анализа данных'):

#-------------график новых/старых клиентов
with st.expander('Вопрос 1: распределение клиентов на "новых" и "старых"'):
	st.plotly_chart(tools.get_newbie_plot(data_train), use_container_width=True)
	radio_newbie = st.radio("Какие выводы можно сделать по данному графику?", 
					(('"Новых" и "старых" клиентов примерно одинаковое количество'), ('Старых клиентов больше'), ('Новых клиентов больше'), ('Затрудняюсь ответить')), index=3)
	if radio_newbie == '"Новых" и "старых" клиентов примерно одинаковое количество':
		st.success(f'*Ответ верный!* Отношение новых клиентов к старым: {(data_train["newbie"] == 1).sum() / (data_train["newbie"] == 0).sum():.2f}')
	else:
		st.error('*Ответ неверный*') 

#-------------график распределения по месту жительства
with st.expander('Вопрос 2: распределение клиентов по месту жительства'):	
	tmp_res = data_train.zip_code.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_zipcode_plot(data_train), use_container_width=True)
	radio_zip = st.radio("Какие выводы можно сделать по данному графику?", 
					(('Клиентов примерно одинаковое количество'), ('Большинство клиентов из пригорода'), ('Старых клиентов больше'), ('Затрудняюсь ответить')), index=3)
	if radio_zip == 'Большинство клиентов из пригорода':
		st.success(f'*Ответ верный!* Из пригорода {tmp_res["Surburban"]:.2f}% клиентов, из города {tmp_res["Urban"]:.2f}% и из села {tmp_res["Rural"]:.2f}%')
	else:
		st.error('*Ответ неверный*') 
	# st.write(f'Большинство клиентов из пригорода: {tmp_res["Surburban"]:.2f}%, из города: {tmp_res["Urban"]:.2f}% и из села: {tmp_res["Rural"]:.2f}%')

with st.expander('Вопрос 3: распределение по каналам сбыта'):
	tmp_res = data_train.channel.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_channel_plot(data_train), use_container_width=True)
	radio_channel = st.radio("Какие выводы можно сделать по данному графику?", 
					(('Клиенты предпочитают покупать через сайт'), ('Большинство клиентов пользуются телефонами'), ('В прошлом году заказов через телефон и сайт было примерно одинаково'), ('Затрудняюсь ответить')), index=3)
	if radio_channel == 'В прошлом году заказов через телефон и сайт было примерно одинаково':
		st.success(f'*Ответ верный!* Через телефон и сайт было {tmp_res["Phone"]:.2f}% и {tmp_res["Web"]:.2f}% заказов соответственно, а {tmp_res["Multichannel"]:.2f}% клиентов покупали товары, воспользовавшись двумя платформами')
	else:
		st.error('*Ответ неверный*') 

with st.expander('Вопрос 4: распределение клиентов по сегментам денежных трат'):	
	tmp_res = data_train.history_segment.value_counts(normalize=True) * 100
	st.plotly_chart(tools.get_history_segment_plot(data_train), use_container_width=True)
	radio_history_segment = st.radio("Какие выводы можно сделать по данному графику?", 
					(('Большинство клиентов тратят до 100$'), ('На графике мы видим нормальное распределение'), ('Богатых клиентов мало'), ('Затрудняюсь ответить')), index=3)
	if radio_history_segment == 'Большинство клиентов тратят до 100$':
		st.success(f'''*Ответ верный!* Большинство пользователей относится к сегменту \$0-\$100 ({tmp_res[0]:.2f}%), второй и третий по количеству пользователей сегменты \$100-\$200 ({tmp_res[1]:.2f}%) и \$200-\$350 ({tmp_res[2]:.2f}%). 
		К сегментам \$350-\$500 и \$500-\$750 относится {tmp_res[3]:.2f}% и {tmp_res[4]:.2f}% пользователей соответственно. Меньше всего пользователей в сегментах \$750-\$1.000 ({tmp_res[-2]:.2f}%) и \$1.000+ ({tmp_res[-1]:.2f}%).''')
	else:
		st.error('*Ответ неверный*') 

with st.expander('Вопрос 5: распределение клиентов по количеству месяцев с последней покупки'):	
	tmp_res = list(data_train.recency.value_counts(normalize=True) * 100)
	st.plotly_chart(tools.get_recency_plot(data_train), use_container_width=True)
	radio_recency = st.radio("Какие выводы можно сделать по данному графику?", 
					(('На графике мы видим убывающий тренд продаж'), ('Клиенты совершают покупки сезонно, раз в полгода'), ('Большинство клиентов совершают покупки часто, не реже раза в месяц'), ('Затрудняюсь ответить')), index=3)
	if radio_recency == 'Большинство клиентов совершают покупки часто, не реже раза в месяц':
		st.success(f'''*Ответ верный!* Большинство клиентов являются активными клиентами платформы: {tmp_res[0]:.2f}% человек совершали покупки в течение месяца. Также заметно, что много клиентов совершали покупки через 9 и 10 месяцев. 
		Это может свидетельствовать о проведении рекламной кампании в это время или чего-то еще: обратите внимание на доли новых клиентов в данном распределении!''')
	else:
		st.error('*Ответ неверный*') 

with st.expander('Вопрос 6: распределение клиентов по сумме потраченных денег за год'):	
	st.plotly_chart(tools.get_history_plot(data_train), use_container_width=True)
	st.markdown('_График интерактивный. Двойной клик вернет в начальное состояние._')
	radio_history = st.radio("Какие выводы можно сделать по данному графику?", 
					(('Половина клиентов тратит 1500$ на покупки'), ('Больше 500$ тратят только новые клиенты'), ('Большинство клиентов тратят около 100$ на покупки'), ('Затрудняюсь ответить')), index=3)
	if radio_history == 'Больше 500$ тратят только новые клиенты':
		st.success(f'''*Ответ верный!* А абсолютное большинство клиентов тратят \$25-\$35 на покупки. Также есть и малая доля тех, кто тратит более \$3.000''')
	else:
		st.error('*Ответ неверный*') 

with st.expander('Вопрос 7: влияние прошлой рекламной кампании на "новых" и "старых" клиентов'):
	data_train['treatment'] = treatment_train 
	data_train['target'] = target_train 
	# st.write(data_train.value_counts(['target']))
	# st.write(data_train.shape)
	# st.write(data_train.groupby(['newbie', 'treatment'])['target'].count().reset_index() )
	# st.write(data_train)
	st.plotly_chart(tools.get_last_treatment_plot(data_train), use_container_width=True)
	radio_treatment = st.radio("Какие выводы можно сделать по данному графику?", 
					(('В прошлую рекламную кампанию и "новые", и "старые" клиенты получали рассылку примерно поровну'), ('Рекламы было отправлено слишком много, кампания была убыточна'), ('"Старым" клиентам нужно отправлять больше рекламы, чем "новым"'), ('Затрудняюсь ответить')), index=3)
	if radio_treatment == 'В прошлую рекламную кампанию и "новые", и "старые" клиенты получали рассылку примерно поровну':
		st.success(f'''*Ответ верный!* Остальные выводы поспешно делать по данному графику''')
	else:
		st.error('*Ответ неверный*') 

with st.expander('Вопрос 8: примените фильтры для анализа разницы между клиентами'):
	groupped_table, diff, pivot_table = tools.get_pivot_table(dataset, target, treatment) # 

	st.markdown('''Чтобы понять, как повлияла прошлая рекламная кампания на клиентов, - составьте сводную таблицу, используя нижеуказанные фильтры.
	\n*В чем особенность сводной таблицы?*
	\nОна отбирает все колонки **по вашим фильтрам** и **группирует их по по целевой колонке** ('targ', переход клиента на сайт по ссылке в письме), а после агрегирует значения по количеству ('count') и среднему значению ('mean').
	\nЧем больше разница в колонке 'diff', тем больше разница среднего значения целевой переменной **при одинаковых значениях остальных ячеек в строке**.
	\n*ПРИМЕР:* 
	\nвыберете в фильтрах: 
	\n- тип клиента - все, 
	\n- товары - любые, 
	\n- сегменты клиентов по объему потраченных денег - все, 
	\n- клиентов по почтовому коду - всех, 
	\nа также в пункте "Отфильтровать разницу" выберете "По убыванию" и нажмите кнопку "Посмотреть данных с учётом фильтров".
	\n*Какие выводы мы можем сделать, применив такие фильтры?* 
	\n1. Смотрим первую строку таблицы:
	\nНаибольшая разница (смотрим отранжированноую колонку 'diff') между средним значением по заходу на сайт после рекламной кампании была у **новых покупателей** (колонка 'newbie' = 1), 
	**живущих в сёлах** (колонка 'zip_code' = Rural), **приобретавших женские товары** (колонка 'womens' = 1), **потративших от 750 до 1000 долларов** (колонка 'history_segment' = \$750-\$1000) и 
	ПОЛУЧИВШИХ рекламу (колонка 'treat' = 1) в отличие от *точно таких же клиентов*, но которые НЕ ПОЛУЧИЛИ рекламу в тот период.
	\n2. Отправить новую рекламу целесообразнее всего клиентам, с наибольшей разницей по колонке 'diff'.
	\n3. Изменяя фильтры, можно более точечено исследовать разницу среди потребителей одной группы (одна строка в сводной таблице), получивших и не получивших рекламу в прошлый период.
	\n*Теперь попробуйте сами сформировать разные варианты сводной таблицы, сделав выводы.* 
	''')
	with st.form(key='treatment_filter'):
		st.markdown('''##### *Фильтры для сводной таблицы:*''')

		col1, col2, col3 = st.columns(3)
		filters = {}
		treatment_filter = col1.radio('Была ли ранее отправлена реклама:', options=['Да']) # ['Всё равно', 'Да', 'Нет']
		filters['treatment_filter'] = treatment_filter

		newbie_filter = col2.radio('Тип клиента:', options=['Все', 'Только новые', 'Только старые'])
		filters['newbie_filter'] = newbie_filter

		mens_filter = col3.radio('Товары, которые приобретались:', options=['Любые', 'Мужские', 'Женские'])
		filters['mens_filter'] = mens_filter

		filters['history_segments'] = {}
		col1, col2, col3 = st.columns(3)
		with col1:
			st.write('Сегмент клиентов по объему денег, потраченных в прошлом году (history segments):')
			first_group = st.checkbox('$0-$100', value=True)
			if first_group:
				filters['history_segments']['1) $0 - $100'] = True
			second_group = st.checkbox('$100-$200', value=True)
			if second_group:
				filters['history_segments']['2) $100 - $200'] = True
			third_group = st.checkbox('$200-$350', value=True)
			if third_group:
				filters['history_segments']['3) $200 - $350'] = True
			fourth_group = st.checkbox('$350-$500', value=True)
			if fourth_group:
				filters['history_segments']['4) $350 - $500'] = True
			fifth_group = st.checkbox('$500-$750', value=True)
			if fifth_group:
				filters['history_segments']['5) $500 - $750'] = True
			sixth_group = st.checkbox('$750-$1.000', value=True)
			if sixth_group:
				filters['history_segments']['6) $750 - $1,000'] = True
			seventh_group = st.checkbox('$1.000+', value=True)
			if seventh_group:
				filters['history_segments']['7) $1,000 +'] = True

		with col2:
			st.write('Каких пользователей по почтовому коду выберем:')
			filters['zip_code'] = {}
			surburban = st.checkbox('Surburban', value=True)
			if surburban:
				filters['zip_code']['surburban'] = True
			urban = st.checkbox('Urban', value=True)
			if urban:
				filters['zip_code']['urban'] = True
			rural = st.checkbox('Rural', value=True)
			if rural:
				filters['zip_code']['rural'] = True
		with col3:
			asc = st.radio('Отфильтровать разницу:', options=['Не фильтровать', 'По возрастанию', 'По убыванию'], index=0)	

		if st.form_submit_button('Посмотреть данные с учётом фильтров'):
			pivot_table = tools.treatment_filter_data(pivot_table, filters)

			if asc == 'Не фильтровать':
				pivot_table = pivot_table.sort_values(by = ['history_segment'], ascending = True)				
			elif asc == 'По возрастанию':
				pivot_table = pivot_table.sort_values(by = ['diff'], ascending = True)	
			else:
				pivot_table = pivot_table.sort_values(by = ['diff'], ascending = False)	


			# pivot_table['№'] = pivot_table.index
			# # filtered_dataset['uplift'] = uplift
			# # uplift_percentile_table['income'] = [filtered_dataset.sort_values('uplift').iloc[int(len(uplift_percentile_table) * (0.1 * i)):int(len(uplift_percentile_table) * 0.1 * (i + 1))].spend.mean() for i in range(10)]
			# cols = ['percentiles', 'n_treatment', 'n_control', 'response_rate_treatment', 'response_rate_control', 'uplift'] #, 'income']
			# uplift_percentile_table = uplift_percentile_table[cols]
			# uplift_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']] = uplift_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']].apply(lambda x: round(x, 4))
			fig = go.Figure(data=[go.Table(
											columnorder = [n for n in range(1, len(pivot_table.columns)+1)],
											columnwidth = [1,1,0.5,0.6,0.6,0.5,0.5,0.6,0.6], # *len(title_subsample.columns),
											header=dict(values=list(pivot_table.columns),
														line_color='black',
														fill_color=['#d9d9d9', '#d9d9d9','#d9d9d9','#d9d9d9','#d9d9d9','#d9d9d9','#d9d9d9','#d9d9d9','#ef553b'],
														align=['center']*len(pivot_table.columns),
														font=dict(color='black', size=14),
														height=30),                                            
											cells=dict(values=[pivot_table[col] for col in pivot_table.columns],
														fill_color=['white', 'white','white','white','white','white','white','white','#e78a83'],
														align=['center']*len(pivot_table.columns),
														font=dict(color='black', size=13),
														height=30)) 
										])
			fig.update_traces(cells_line_color='black')
			fig.update_layout(width=640, margin=dict(b=0, l=0, r=1, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,  , height=245
			st.plotly_chart(fig)
			

			


#-----------------------------БЛОК 2 РУЧНАЯ ФИЛЬТРАЦИЯ-----------------------------#
st.header('Блок 2: Самостоятельный выбор клиентов для отправки рекламы', anchor='student_choise')
st.markdown('''Примените фильтры ниже для того, чтобы вручную отобрать клиентов, которые попадут в маркетинговую акцию''')
filters = {}
with st.form(key='filter-clients'):
	st.markdown('''##### *Кому отправить рекламу?*''')

	col1, col2, col3 = st.columns(3)

	channel_filter = col1.radio('Канал покупки прошлом году', options=['Все', 'Phone', 'Web', 'Multichannel'])
	filters['channel_filter'] = channel_filter

	newbie_filter = col2.radio('Тип клиента', options=['Все', 'Только новые', 'Только старые'])
	filters['newbie_filter'] = newbie_filter

	mens_filter = col3.radio('Клиенты, приобретавшие товары', options=['Любые', 'Мужские', 'Женские'])
	filters['mens_filter'] = mens_filter

	filters['history_segments'] = {}
	col1, col2 = st.columns(2)
	with col1:
		st.write('Класс клиентов по объему денег, потраченных в прошлом году (history segments)')
		first_group = st.checkbox('$0-$100', value=True)
		if first_group:
			filters['history_segments']['1) $0 - $100'] = True
		second_group = st.checkbox('$100-$200', value=True)
		if second_group:
			filters['history_segments']['2) $100 - $200'] = True
		third_group = st.checkbox('$200-$350', value=True)
		if third_group:
			filters['history_segments']['3) $200 - $350'] = True
		fourth_group = st.checkbox('$350-$500', value=True)
		if fourth_group:
			filters['history_segments']['4) $350 - $500'] = True
		fifth_group = st.checkbox('$500-$750', value=True)
		if fifth_group:
			filters['history_segments']['5) $500 - $750'] = True
		sixth_group = st.checkbox('$750-$1.000', value=True)
		if sixth_group:
			filters['history_segments']['6) $750 - $1,000'] = True
		seventh_group = st.checkbox('$1.000+', value=True)
		if seventh_group:
			filters['history_segments']['7) $1,000 +'] = True

	with col2:
		st.write('Каких пользователей по почтовому коду выберем')
		filters['zip_code'] = {}
		surburban = st.checkbox('Surburban', value=True)
		if surburban:
			filters['zip_code']['surburban'] = True
		urban = st.checkbox('Urban', value=True)
		if urban:
			filters['zip_code']['urban'] = True
		rural = st.checkbox('Rural', value=True)
		if rural:
			filters['zip_code']['rural'] = True

	recency = st.slider(label='Месяцев с момента покупки', min_value=int(data_test.recency.min()), max_value=int(data_test.recency.max()), value=(int(data_test.recency.min()), int(data_test.recency.max())))
	filters['recency'] = recency

	st.write('Если известно на какой процент аудитории необходимо повлиять, измените значение')
	target_volume = st.slider(label='Процент аудитории', min_value=1, max_value=100, value=100)
	# k = k/100-0.0001

	filter_form_submit_button = st.form_submit_button('Применить фильтр')

# проверка корректности заполнения форм
if not first_group and not second_group and not third_group and not fourth_group and not fifth_group and not sixth_group and not seventh_group:
	st.error('Необходимо выбрать хотя бы один класс')
	st.stop()
elif not surburban and not urban and not rural:
	st.error('Необходимо выбрать хотя бы один почтовый индекс')
	st.stop()

# фильтруем тестовые данные по пользовательскому выбору
filtered_dataset = tools.filter_data(data_test, filters)

# проверяем, что данные отфильтровались
if filtered_dataset is None:
	st.error('Не найдено пользователей для данных фильтров. Попробуйте изменить фильтры.')
	st.stop()

# значение uplift для записей тех клиентов, который выбрал пользователь равен 1
uplift = [1 for _ in filtered_dataset.index]
target_filtered = target_test.loc[filtered_dataset.index]
treatment_filtered = treatment_test.loc[filtered_dataset.index]

# блок с демонстрацией отфильтрованных данных
with st.expander(label='Посмотреть пример пользователей, которым будет отправлена реклама'):
	sample_size = 7 if filtered_dataset.shape[0] >= 7 else filtered_dataset.shape[0]
	example = filtered_dataset.sample(sample_size)

	fig = go.Figure(data=[go.Table(
                                columnorder = [n for n in range(1, len(example.columns)+1)],
                                columnwidth = [1,1.7,1,1,1,1.2,1,1.3,1], # *len(example.columns),
                                header=dict(values=list(example.columns),
                                            line_color='black',
                                            fill_color='#d9d9d9',
                                            align=['center']*len(example.columns),
                                            font=dict(color='black', size=14),
                                            height=30),                                            
                                cells=dict(values=[example[col] for col in example.columns],
                                            fill_color='white',
                                            align=['center']*len(example.columns),
                                            font=dict(color='black', size=14),
                                            height=30))
                              ])
	fig.update_traces(cells_line_color='black')
	fig.update_layout(width=670, height=245, margin=dict(b=0, l=0, r=1, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
	st.plotly_chart(fig)
	# st.dataframe(example)
	st.info(f'''Количество пользователей, попавших в выборку: {filtered_dataset.shape[0]} ({filtered_dataset.shape[0] / data_test.shape[0] * 100 :.2f}%) ''')
	# \n Суммарно они потратили в прошлом году {filtered_dataset.history.sum() :.2f}''')
	res = st.button('Обновить')

#------------------------------------------------------------------#
with st.expander('Результаты ручной фильтрации', expanded=True):
	# считаем метрики для пользователя
	user_metric_uplift_at_k = uplift_at_k(target_filtered, uplift, treatment_filtered, strategy='overall', k=target_volume)
	user_metric_uplift_by_percentile = uplift_by_percentile(target_filtered, uplift, treatment_filtered)
	user_metric_qini_auc_score = qini_auc_score(target_filtered, uplift, treatment_filtered)
	user_metric_weighted_average_uplift = tools.get_weighted_average_uplift(target_filtered, uplift, treatment_filtered)
	qini_curve_user_score = qini_curve(target_filtered, uplift, treatment_filtered)
	uplift_curve_user_score = uplift_curve(target_filtered, uplift, treatment_filtered)
	# отображаем метрики
	col1, col2, col3 = st.columns(3)
	col1.metric(label=f'Uplift для {target_volume}% пользователей', value=f'{user_metric_uplift_at_k:.4f}') # int(round(k*100, 0)
	col2.metric(label=f'Qini AUC score', value=f'{user_metric_qini_auc_score:.4f}', help='Всегда будет 0 для пользователя')
	col3.metric(label=f'Weighted average uplift', value=f'{user_metric_weighted_average_uplift:.4f}')

	st.write('Uplift по процентилям')
	# строим таблицу через plotly:
	uplift_percentile_table = pd.DataFrame(user_metric_uplift_by_percentile)
	uplift_percentile_table['percentiles'] = uplift_percentile_table.index
	# filtered_dataset['uplift'] = uplift
	# uplift_percentile_table['income'] = [filtered_dataset.sort_values('uplift').iloc[int(len(uplift_percentile_table) * (0.1 * i)):int(len(uplift_percentile_table) * 0.1 * (i + 1))].spend.mean() for i in range(10)]
	cols = ['percentiles', 'n_treatment', 'n_control', 'response_rate_treatment', 'response_rate_control', 'uplift'] #, 'income']
	uplift_percentile_table = uplift_percentile_table[cols]
	uplift_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']] = uplift_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']].apply(lambda x: round(x, 4))

	
	fig = go.Figure(data=[go.Table(
                                columnorder = [n for n in range(1, len(uplift_percentile_table.columns)+1)],
                                columnwidth = [1,1,1,2,2,1], # *len(example.columns),
                                header=dict(values=list(uplift_percentile_table.columns),
                                            line_color='black',
                                            fill_color='#d9d9d9',
                                            align=['center']*len(uplift_percentile_table.columns),
                                            font=dict(color='black', size=14),
                                            height=30),                                            
                                cells=dict(values=[uplift_percentile_table[col] for col in uplift_percentile_table.columns], 
                                            fill_color='white',
                                            align=['center']*len(uplift_percentile_table.columns),
                                            font=dict(color='black', size=14),
                                            height=30))
                              ])
	fig.update_traces(cells_line_color='black')
	fig.update_layout(width=670, height=340, margin=dict(b=0, l=0, r=2, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
	st.plotly_chart(fig)
	# st.write(user_metric_uplift_by_percentile)
	

# #-----------------------------БЛОК 3-----------------------------#
st.header('Блок 3: Подбор стратегии с помощью ML', anchor='ml_models')
st.markdown('''#### Задание по Блоку 3:
\nВ этом задании вы сравните свой отбор клиентов из предыдущего блока с тем, как это делают модели машинного обучения: CatBoost, RandomForestClassifier и XGBoost.
\n##### *Как это делать?*
\nПеред вами внизу три раскрывающихся вкладки. Открыв каждую, вы увидите график, сравнивающий работу соответствующей модели машинного обучения по отбору клиенов, и ваш самостоятельный выбор из Блока 2.
\nНа каждом из графике будет отображено по 3 элемента:
\n1. Синяя AUC-кривая - это график зависимости дохода от рекламной кампании (ось Y) от количества клиентов, получивших рекламу (ось X), которую выявила соответствующая модель машинного обучения;
\n2. Красная точка ("Analitic qini") - это результат вашего ручного отбора из из Блока 2: точка указывает, какой доход (ось Y) вы получите, если отправите рекламу всем клиентам, которых выбрали вручную (их количество - по оси X);
\n3. Оранжевая диагональная прямая ("Random") обозначает решение задачи по отбору клиентов методом случайного выбора.
\n##### *Как это понимать?*
\nКонечно, не всегда бОльшие раходы на рекламу означают бОльшую прибыль после неё. И, естественно, нам надо найти оптимальное соотношение между расходами на кампанию (количество клиентов, что получат реклау, ось X) и доходом от этого (ось Y).
\n**Для этого, оцените зависимость "доходы-количество клиентов, получивших рекламу" на пиках AUC-кривой и сделайте соответствующие выводы по работе каждой модели.**
\nЕщё несколько моментов:
\n1. Если ваша точка ручного отбора ("Analitic qini") лежит *выше* синей AUC-кривой - то это тот редкий случай, когда ручной выбор аналитика сработал лучше математической модели;
\n2. Если точка "Analitic qini" лежит *между* синей AUC-кривой и оранжевой прямой - это значит, что вы отобрали клиентов лучше, чем случаный отбор, но хуже, чем модель машинного обучения;
\n3. Если же она располжена *ниже*, чем прямая "Random", то ваши критерии выбора клиентов совсем не оптимальны и даже хуже, чем определение их через подбрасывание монетки.
''')
# show_ml_reasons = st.checkbox('Показать решения с помощью ML')
# if show_ml_reasons:
with st.expander('Решение с помощью CatBoost'):
	with st.form(key='catboost_metricks'):

		final_uplift = sm_cbc.loc[filtered_dataset.index]['0']

		# считаем метрики для ML
		catboost_uplift_at_k = uplift_at_k(target_filtered, final_uplift, treatment_filtered, strategy='overall', k=target_volume)
		catboost_uplift_by_percentile = uplift_by_percentile(target_filtered, final_uplift, treatment_filtered)
		catboost_qini_auc_score = qini_auc_score(target_filtered, final_uplift, treatment_filtered)
		catboost_weighted_average_uplift = tools.get_weighted_average_uplift(target_filtered, final_uplift, treatment_filtered)

		# отображаем метрики
		col1, col2, col3 = st.columns(3)
		col1.metric(
			label=f'Uplift для {target_volume}% пользователей',
			value=f'{catboost_uplift_at_k:.4f}',
			delta=f'{catboost_uplift_at_k - user_metric_uplift_at_k:.4f}'
		)
		col2.metric(
			label=f'Qini AUC score',
			value=f'{catboost_qini_auc_score:.4f}',
			help='Всегда будет 0 для пользователя',
			delta=f'{catboost_qini_auc_score - user_metric_qini_auc_score:.4f}'
		)
		col3.metric(
			label=f'Weighted average uplift',
			value=f'{catboost_weighted_average_uplift:.4f}',
			delta=f'{catboost_weighted_average_uplift - user_metric_weighted_average_uplift:.4f}'
		)

		st.write('Uplift по процентилям')

		catboost_percentile_table = pd.DataFrame(catboost_uplift_by_percentile)
		catboost_percentile_table['percentiles'] = catboost_percentile_table.index
		cols = ['percentiles', 'n_treatment', 'n_control', 'response_rate_treatment', 'response_rate_control', 'uplift']
		catboost_percentile_table = catboost_percentile_table[cols]
		catboost_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']] = catboost_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']].apply(lambda x: round(x, 4))
		
		fig = go.Figure(data=[go.Table(
									columnorder = [n for n in range(1, len(catboost_percentile_table.columns)+1)],
									columnwidth = [1,1,1,2,2,1], # *len(example.columns),
									header=dict(values=list(catboost_percentile_table.columns),
												line_color='black',
												fill_color='#d9d9d9',
												align=['center']*len(catboost_percentile_table.columns),
												font=dict(color='black', size=14),
												height=30),                                            
									cells=dict(values=[catboost_percentile_table[col] for col in catboost_percentile_table.columns], 
												fill_color='white',
												align=['center']*len(catboost_percentile_table.columns),
												font=dict(color='black', size=14),
												height=30))
								])
		fig.update_traces(cells_line_color='black')
		fig.update_layout(width=640, height=340, margin=dict(b=0, l=0, r=2, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
		st.plotly_chart(fig)		
		# st.write(catboost_uplift_by_percentile)
		st.form_submit_button('Обновить графики', help='При изменении флагов')

		perfect_qini = st.checkbox('Отрисовать идеальную метрику qini')
		# получаем координаты пользовательской метрики для точки на графике
		x, y = qini_curve_user_score[0][1], qini_curve_user_score[1][1]
		# получаем объект UpliftCurveDisplay с осями и графиком matplotlib
		qini_fig = plot_qini_curve(target_test, sm_cbc['0'], treatment_test, perfect=perfect_qini) #, color='green')
		# plt.rcParams.update({'axes.facecolor':'white'})
		# qini_fig.ax_.set_facecolor('white')
		# добавляем пользовательскую метрику на оси графика
		qini_fig.ax_.plot(x, y, 'ro', markersize=3, label='Analitic qini')
		# добавляем обозначение метрики пользователя в легенду
		qini_fig.ax_.legend(loc=u'upper left', bbox_to_anchor=(0.75, 0.25))
		# f1 = go.Figure(qini_fig)
		# qini_fig.update_layout(width=640, height=340, margin=dict(b=0, l=0, r=2, t=0))
		st.plotly_chart(qini_fig.figure_, use_container_width=True)
		# st.pyplot(qini_fig.figure_)

		prefect_uplift = st.checkbox('Отрисовать идеальную метрику uplift')
		# получаем координаты пользовательской метрики для точки на графике
		x, y = uplift_curve_user_score[0][1], uplift_curve_user_score[1][1]
		# получаем объект UpliftCurveDisplay с осями и графиком matplotlib
		uplift_fig = plot_uplift_curve(target_test, sm_cbc['0'], treatment_test, perfect=prefect_uplift)
		# добавляем пользовательскую метрику на оси графика
		uplift_fig.ax_.plot(x, y, 'ro', markersize=3, label='Analitic qini')
		# добавляем обозначение метрики пользователя в легенду
		uplift_fig.ax_.legend(loc=u'upper left', bbox_to_anchor=(0.75, 0.25))
		st.plotly_chart(uplift_fig.figure_, use_container_width=True)
		# st.pyplot(uplift_fig.figure_)

with st.expander('Решение с помощью RandomForestClassifier'):
	with st.form(key='sklearn_metricks'):

		final_rf_uplift = tm_rfc.loc[filtered_dataset.index]['0']

		# считаем метрики для ML
		random_forest_uplift_at_k = uplift_at_k(target_filtered, final_rf_uplift, treatment_filtered, strategy='overall', k=target_volume)
		random_forest_uplift_by_percentile = uplift_by_percentile(target_filtered, final_rf_uplift, treatment_filtered)
		random_forest_qini_auc_score = qini_auc_score(target_filtered, final_rf_uplift, treatment_filtered)
		random_forest_weighted_average_uplift = tools.get_weighted_average_uplift(target_filtered, final_rf_uplift, treatment_filtered)

		# отображаем метрики
		col1, col2, col3 = st.columns(3)
		col1.metric(
			label=f'Uplift для {target_volume}% пользователей',
			value=f'{random_forest_uplift_at_k:.4f}',
			delta=f'{random_forest_uplift_at_k - user_metric_uplift_at_k:.4f}'
		)
		col2.metric(
			label=f'Qini AUC score',
			value=f'{random_forest_qini_auc_score:.4f}',
			help='Всегда будет 0 для пользователя',
			delta=f'{random_forest_qini_auc_score - user_metric_qini_auc_score:.4f}'
		)
		col3.metric(
			label=f'Weighted average uplift',
			value=f'{random_forest_weighted_average_uplift:.4f}',
			delta=f'{random_forest_weighted_average_uplift - user_metric_weighted_average_uplift:.4f}'
		)

		st.write('Uplift по процентилям')
		random_forest_percentile_table = pd.DataFrame(random_forest_uplift_by_percentile)
		random_forest_percentile_table['percentiles'] = random_forest_percentile_table.index
		cols = ['percentiles', 'n_treatment', 'n_control', 'response_rate_treatment', 'response_rate_control', 'uplift']
		random_forest_percentile_table = random_forest_percentile_table[cols]
		random_forest_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']] = random_forest_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']].apply(lambda x: round(x, 4))
		
		fig = go.Figure(data=[go.Table(
									columnorder = [n for n in range(1, len(random_forest_percentile_table.columns)+1)],
									columnwidth = [1,1,1,2,2,1], # *len(example.columns),
									header=dict(values=list(random_forest_percentile_table.columns),
												line_color='black',
												fill_color='#d9d9d9',
												align=['center']*len(random_forest_percentile_table.columns),
												font=dict(color='black', size=14),
												height=30),                                            
									cells=dict(values=[random_forest_percentile_table[col] for col in random_forest_percentile_table.columns], 
												fill_color='white',
												align=['center']*len(random_forest_percentile_table.columns),
												font=dict(color='black', size=14),
												height=30))
								])
		fig.update_traces(cells_line_color='black')
		fig.update_layout(width=640, height=340, margin=dict(b=0, l=0, r=2, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
		st.plotly_chart(fig)
		# st.write(random_forest_uplift_by_percentile)

		st.form_submit_button('Обновить графики', help='При изменении флагов')

		perfect_qini = st.checkbox('Отрисовать идеальную метрику qini')
		# получаем координаты пользовательской метрики для точки на графике
		x, y = qini_curve_user_score[0][1], qini_curve_user_score[1][1]
		# получаем объект UpliftCurveDisplay с осями и графиком matplotlib
		qini_fig = plot_qini_curve(target_test, tm_rfc['0'], treatment_test, perfect=perfect_qini)
		# добавляем пользовательскую метрику на оси графика
		qini_fig.ax_.plot(x, y, 'ro', markersize=3, label='Analitic qini')
		# добавляем обозначение метрики пользователя в легенду
		qini_fig.ax_.legend(loc=u'upper left', bbox_to_anchor=(0.75, 0.25))
		st.plotly_chart(qini_fig.figure_, use_container_width=True)
		# st.pyplot(qini_fig.figure_)

		prefect_uplift = st.checkbox('Отрисовать идеальную метрику uplift')
		# получаем координаты пользовательской метрики для точки на графике
		x, y = uplift_curve_user_score[0][1], uplift_curve_user_score[1][1]
		# получаем объект UpliftCurveDisplay с осями и графиком matplotlib
		uplift_fig = plot_uplift_curve(target_test, tm_rfc['0'], treatment_test, perfect=prefect_uplift)
		# добавляем пользовательскую метрику на оси графика
		uplift_fig.ax_.plot(x, y, 'ro', markersize=3, label='Analitic qini')
		# добавляем обозначение метрики пользователя в легенду
		uplift_fig.ax_.legend(loc=u'upper left', bbox_to_anchor=(0.75, 0.25))
		st.plotly_chart(uplift_fig.figure_, use_container_width=True)
		# st.pyplot(uplift_fig.figure_)
		print('some')

with st.expander('Решение с помощью XGBoost'):
	with st.form(key='xgboost_metricks'):

		final_xgboost_uplift = sm_xgboost.loc[filtered_dataset.index]['0']

		# считаем метрики для ML
		xgboost_uplift_at_k = uplift_at_k(target_filtered, final_xgboost_uplift, treatment_filtered, strategy='overall', k=target_volume)
		xgboost_uplift_by_percentile = uplift_by_percentile(target_filtered, final_xgboost_uplift, treatment_filtered)
		xgboost_qini_auc_score = qini_auc_score(target_filtered, final_xgboost_uplift, treatment_filtered)
		xgboost_weighted_average_uplift = tools.get_weighted_average_uplift(target_filtered, final_xgboost_uplift, treatment_filtered)

		# отображаем метрики
		col1, col2, col3 = st.columns(3)
		col1.metric(
			label=f'Uplift для {target_volume}% пользователей',
			value=f'{xgboost_uplift_at_k:.4f}',
			delta=f'{xgboost_uplift_at_k - user_metric_uplift_at_k:.4f}'
		)
		col2.metric(
			label=f'Qini AUC score',
			value=f'{xgboost_qini_auc_score:.4f}',
			help='Всегда будет 0 для пользователя',
			delta=f'{xgboost_qini_auc_score - user_metric_qini_auc_score:.4f}'
		)
		col3.metric(
			label=f'Weighted average uplift',
			value=f'{xgboost_weighted_average_uplift:.4f}',
			delta=f'{xgboost_weighted_average_uplift - user_metric_weighted_average_uplift:.4f}'
		)

		st.write('Uplift по процентилям')
		xgboost_percentile_table = pd.DataFrame(xgboost_uplift_by_percentile)
		xgboost_percentile_table['percentiles'] = xgboost_percentile_table.index
		cols = ['percentiles', 'n_treatment', 'n_control', 'response_rate_treatment', 'response_rate_control', 'uplift']
		xgboost_percentile_table = xgboost_percentile_table[cols]
		xgboost_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']] = xgboost_percentile_table[['response_rate_treatment', 'response_rate_control','uplift']].apply(lambda x: round(x, 4))
		
		fig = go.Figure(data=[go.Table(
									columnorder = [n for n in range(1, len(xgboost_percentile_table.columns)+1)],
									columnwidth = [1,1,1,2,2,1], # *len(example.columns),
									header=dict(values=list(xgboost_percentile_table.columns),
												line_color='black',
												fill_color='#d9d9d9',
												align=['center']*len(xgboost_percentile_table.columns),
												font=dict(color='black', size=14),
												height=30),                                            
									cells=dict(values=[xgboost_percentile_table[col] for col in xgboost_percentile_table.columns], 
												fill_color='white',
												align=['center']*len(xgboost_percentile_table.columns),
												font=dict(color='black', size=14),
												height=30))
								])
		fig.update_traces(cells_line_color='black')
		fig.update_layout(width=640, height=340, margin=dict(b=0, l=0, r=2, t=0)) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
		st.plotly_chart(fig)
		# st.write(xgboost_uplift_by_percentile)
		st.form_submit_button('Обновить графики', help='При изменении флагов')

		perfect_qini = st.checkbox('Отрисовать идеальную метрику qini')
		# получаем координаты пользовательской метрики для точки на графике
		x, y = qini_curve_user_score[0][1], qini_curve_user_score[1][1]
		# получаем объект UpliftCurveDisplay с осями и графиком matplotlib
		qini_fig = plot_qini_curve(target_test, sm_xgboost['0'], treatment_test, perfect=perfect_qini)
		# добавляем пользовательскую метрику на оси графика
		qini_fig.ax_.plot(x, y, 'ro', markersize=3, label='Analitic qini')
		# добавляем обозначение метрики пользователя в легенду
		qini_fig.ax_.legend(loc=u'upper left', bbox_to_anchor=(0.75, 0.25))
		st.plotly_chart(qini_fig.figure_, use_container_width=True)
		# st.pyplot(qini_fig.figure_)

		prefect_uplift = st.checkbox('Отрисовать идеальную метрику uplift')
		# получаем координаты пользовательской метрики для точки на графике
		x, y = uplift_curve_user_score[0][1], uplift_curve_user_score[1][1]
		# получаем объект UpliftCurveDisplay с осями и графиком matplotlib
		uplift_fig = plot_uplift_curve(target_test, sm_xgboost['0'], treatment_test, perfect=prefect_uplift)
		# добавляем пользовательскую метрику на оси графика
		uplift_fig.ax_.plot(x, y, 'ro', markersize=3, label='Analitic qini')
		# добавляем обозначение метрики пользователя в легенду
		uplift_fig.ax_.legend(loc=u'upper left', bbox_to_anchor=(0.75, 0.25))
		st.plotly_chart(uplift_fig.figure_, use_container_width=True)
		# st.pyplot(uplift_fig.figure_)


#-----------------------------БЛОК 4-----------------------------#
st.header('Блок 4: Подбор стратегии исходя из бюджета', anchor='sreategy_budget')
st.markdown('''#### Задание по Блоку 4:
\nВ данном задании отбирать клиентов для отправки рекламы будет модель из Блока 3 - RandomForestClassifier.
\nА вы, выбирая цену коммуникации и общий бюджет на рекламу, ответьте на следующие вопросы:
\n1. Как зависят прибыль рекламной кампании от дохода после её проведения?
\n2. Где на графике находится т.н. "точка безубыточности"?
\n3. При общем рекламном бюджете в 2000 долларов, на какую максимальную цену за коммуникацию вы можете согласиться, чтобы общая рекламная кампания не была убыточной?
\n4. Если стоимосить отправки рекламы клиенту равна 90 центов, будет ли рационально потратить на рекламу 510 долларов? 1150 долларов? 1700 долларов?
''')
with st.form(key='sreategy_budget'):
	st.markdown('''##### *Условия для графика эластичности:*''')
	# random forest
	communication_cost = st.slider(label='Цена коммуникации в центах:', min_value=0, max_value=200, value=1, step=1) / 100
	total_budget = st.slider(label='Бюджет на рекламу в долларах:', min_value=100, max_value=2000, value=100, step=1)
	# target_volume = total_budget / (communication_cost * 100)

	treatment_mask = (treatment_test == 1)
	rfc_treatment = tm_rfc[treatment_mask]

	# number_top_targets = int(len(rfc_treatment) * target_volume / 100)

	sorted_uplift = rfc_treatment.sort_values('0', ascending=False)

	# top_uplift_index = sorted_uplift[:number_top_targets].index

	# top_uplift_data = data_test.loc[top_uplift_index]

	# total_treatment_cost = communication_cost * number_top_targets
	# total_treatment_spend = top_uplift_data['spend'].sum()

	# total_profit = total_treatment_spend - total_treatment_cost

	# st.write(number_top_targets)
	# st.write(total_treatment_cost)
	# st.write(total_treatment_spend)
	# st.write(total_profit)

	cost_com_tensor = np.linspace(0.1, 2, 100)
	spend_com_tensor = np.linspace(0.1, 2, 100)
	total_cost_tensor = np.linspace(0.1, 2, 100)
	if st.form_submit_button('Построить график'):
		volume_tensor = total_budget / cost_com_tensor
		for i, (target_volume, communication_cost) in enumerate(zip(volume_tensor, cost_com_tensor)):
			# number_top_targets = int(len(rfc_treatment) * target_volume / 100)
			number_top_targets = int(target_volume)
			top_uplift_index = sorted_uplift[:number_top_targets].index
			top_uplift_data = data_test.loc[top_uplift_index]
			total_treatment_cost = communication_cost * number_top_targets
			total_treatment_spend = top_uplift_data['spend'].sum()
			spend_com_tensor[i] = total_treatment_spend
			total_cost_tensor[i] = total_treatment_cost

		# total_cost_tensor = cost_com_tensor * number_top_targets
		profit_com_tensor = spend_com_tensor - total_cost_tensor #


		# profit_com_tensor, total_cost_tensor = tools.compute_profit_spend_cost(communication_cost, target_volume)

		fig = px.line(title='Кривая эластичности') # x=cost_com_tensor, y=profit_com_tensor, labels='cost_com_tensor|profit_com_tensor', 
		fig.add_scatter(x=cost_com_tensor, y=spend_com_tensor, hovertext='Зависимость дохода от цены коммуникации', hovertemplate="Реклама: %{x}$ Доход: %{y}$",  name='Зависимость дохода от цены коммуникации', line_color='orange') # round(float(cost_com_tensor[0]),2)
		fig.add_scatter(x=cost_com_tensor, y=profit_com_tensor, hovertext='Зависимость прибыли от цены коммуникаци', hovertemplate="Реклама: %{x}$ Прибыль: %{y}$", name='Зависимость прибыли от цены коммуникаци', line_color='green', hoverinfo="x+y") # hovertext='Зависимость прибыли от цены коммуникации',
		fig.add_scatter(x=cost_com_tensor, y=total_cost_tensor, hovertext='Бюджет на рекламу', hovertemplate="Потраченый бюджет: %{y}$", name='Потраченый бюджет', line_color='red') # trace1
		# fig.update_traces(cells_line_color='black')
		fig.update_xaxes(title='Цена коммуникации в долларах',
						automargin = True,
						categoryorder='total ascending', # 'total descending'
						range=[0, 2],
						tickvals=np.linspace(0,2,21), #int(year) for year in docs_from_year.keys()], # if docs_from_year.keys().index(year)/10==0],
						autorange = True,
						# tickangle =270,
						# tickfont = dict(size=10)
						)
		fig.update_yaxes(title='Сумма в долларах',
						automargin = True,
						categoryorder='total ascending', # 'total descending'
						tickmode='linear',
						tick0=0,
						dtick=1000,
						autorange = True,
						# nticks=10,
						# range=[-500, 8000],
						# tickvals=np.linspace(-500,8000,18), #int(year) for year in docs_from_year.keys()], # if docs_from_year.keys().index(year)/10==0],
						# tickangle =270,
						# tickfont = dict(size=10)
						)				

		fig.update_layout(showlegend=True,
						legend=dict(
									yanchor="top",
									y=0.99,
									xanchor="right",
									x=0.99
									),
						# legend_traceorder="reversed",
						autosize=True,
			width=670, height=400, margin=dict(b=0, l=0, r=1, t=23),
			) # bottom, left, right и top - отступы     title='Наша таблица', title_x=0.5, title_y=1,
		st.plotly_chart(fig)

