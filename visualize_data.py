import pandas as pd 
from process_data import clean_data
import matplotlib.pyplot as plt

data = clean_data('data_1.csv')
features = ['Temperature_Avg', 'Dew_Point_Avg', 'Humidity_Avg', 'Wind_Speed_Avg', 'Pressure_Avg']

# biểu đồ nhiệt độ
plt.figure(figsize=(12, 8))
# plt.plot('Date', 'Temperature_Avg', data=data, linewidth=1)
plt.bar('Date', 'Temperature_Avg', data=data)
plt.title('Daily temperature graph')
plt.ylabel('Degrees in C')
plt.xlabel('Date')
plt.show()

plot_features = data[features]
plot_features.index = data.Date

fig, axes = plt.subplots(
    nrows=2, 
    ncols=1, 
    figsize=(15, 10), 
    facecolor="w", 
    edgecolor="k"
)

for i, feature in enumerate(['Pressure_Avg', 'Wind_Speed_Avg']):
    axes[i % 2].plot(plot_features[feature])
    axes[i % 2].set_title(f'{feature} hanoi - hourly')  
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(
    nrows=3, 
    ncols=1, 
    figsize=(15, 10), 
    facecolor="w", 
    edgecolor="k"
)
for i , feature in enumerate(['Temperature_Avg', 'Dew_Point_Avg', 'Humidity_Avg']):
    axes[i % 3].plot(plot_features[feature])
    axes[i % 3].set_title(f'{feature} hanoi - daily')  
plt.tight_layout()
plt.show()


plot_features = data.groupby('date')[features].mean()
fig, axes = plt.subplots(
    nrows=2, 
    ncols=2, 
    figsize=(15, 10), 
    facecolor="w", 
    edgecolor="k"
)
for i, feature in enumerate(['Temperature_Avg', 'Dew_Point_Avg', 'Humidity_Avg', 'Wind_Speed_Avg']):
    axes[i // 2, i % 2].plot(plot_features[feature])
    axes[i // 2, i % 2].set_title(f'{feature} HaNoi - daily')
        
plt.tight_layout()
plt.show()

# plt.figure(figsize=(8, 8))
# plt.hist2d(data['Pressure_Avg'], data['Temperature_Avg'], bins=(50, 50))
# plt.colorbar()
# ax = plt.gca()
# plt.xlabel('Pressure_Avg, hPa')
# plt.ylabel('Temperature_Avg, C')
# ax.axis('tight')
# plt.show()


# plt.figure(figsize=(8, 8))
# plt.hist2d(data['Wind_Speed_Avg'], data['Temperature_Avg'], bins=(50, 50))
# plt.colorbar()
# ax = plt.gca()
# plt.xlabel('Wind speed, m/s')
# plt.ylabel('Temperature, C')
# ax.axis('tight')
# plt.show()


data['month'] = [x.month for x in data['Date']]
data.boxplot('Temperature_Avg', by='month', figsize=(12, 8), grid=False)
plt.show()