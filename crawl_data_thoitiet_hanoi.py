from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from selenium.webdriver.chrome.options import Options

# Tạo tùy chọn trình duyệt
options = Options()
options.add_argument('--headless')  # Kích hoạt chế độ headless

# Khởi tạo trình điều khiển trình duyệt
driver = webdriver.Chrome(options=options)

# columns=['Max_T','Avg_T','Min_T','Max_D','Avg_D','Min_D','Max_H','Avg_H','Min_H','Max_W','Avg_W','Min_W','Max_P','Avg_P','Min_P','Total']
# cl = ['Year', 'Month', 'Day', 'Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed','Wind Gust', 'Pressure', 'Precip.', 'Condition']
df = pd.DataFrame() 

for year in range(2010, 2022):
    for month in range(1, 13):
        year_month = str(year) + "-" + str(month) 
        print(year_month)
        url = 'https://www.wunderground.com/history/monthly/vn/hanoi/VVNB/date/'+ year_month
        
        driver.get(url)
        tables = WebDriverWait(driver,20).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table")))
        df_table = pd.DataFrame()
        for i in range(2, len(tables)):
            newTable = pd.read_html(tables[i].get_attribute('outerHTML'))
            df_table = pd.concat([df_table, newTable[0]], axis = 1, ignore_index=True)
            # print(df_table)
        month_year = pd.DataFrame({'Year': [int(year)]*(len(df_table)),'Month': [int(month)]*(len(df_table))})
        df_table = pd.concat([month_year, df_table] , axis=1, ignore_index=True)
        df_table = df_table.drop(0)
        df = pd.concat([df, df_table], ignore_index=True)
        df.to_csv('data_1.csv', index = False)

        
                        