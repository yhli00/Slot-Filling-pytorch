# 实验一
参数
加上了ent_embedding和pos_embedding
| 参数 | 参数值 |
| ---- | ----- |
|框架| paddle |
|model|bert-large-uncased|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|100|
|n_top|5|
| batch_size | 8|
|context_max_len|24|
|label_max_len|16|
|显卡|3090|

结果
|domain|baseline(f1)|my_model(f1)|
|---|---|---|
|AddToPlaylist| 0.6870 |0.6291|
|BookRestaurant| 0.6349 |0.4120|
|GetWeather| 0.6536|0.6232|
|PlayMusic| 0.5351| 0.4539|
|RateBook| 0.3651| 0.3149|
|SearchCreativeWork| 0.6922| 0.7711|
|SearchScreeningEvent|0.3354|0.3905|
|avg|0.5576|0.5135|

# 实验二
在实验一的基础上去掉了ent_embedding和pos_embedding，其余参数没变
| 参数 | 参数值 |
| ---- | ----- |
|框架| paddle |
|model|bert-large-uncased|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|100|
|n_top|5|
| batch_size | 8|
|context_max_len|24|
|label_max_len|16|
|显卡|3090|

结果
|domain|baseline(f1)|my_model(f1)|
|---|---|---|
|AddToPlaylist| 0.6870 |0.5667|
|BookRestaurant| 0.6349 |0.3869|
|GetWeather| 0.6536|0.6194|
|PlayMusic| 0.5351| 0.4705|
|RateBook| 0.3651| 0.3176|
|SearchCreativeWork| 0.6922| 0.6919|
|SearchScreeningEvent|0.3354|0.3822|
|avg|0.5576|0.4907|

# 实验三
在实验一的基础上把框架改成了torch
| 参数 | 参数值 |
| ---- | ----- |
|框架| pytorch |
|model|bert-large-uncased|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|100|
|n_top|5|
| batch_size | 8|
|context_max_len|24|
|label_max_len|16|
|显卡|3090|

结果
|domain|baseline(f1)|my_model(f1)|
|---|---|---|
|AddToPlaylist| 0.6870 |0.6149|
|BookRestaurant| 0.6349 |0.3372|
|GetWeather| 0.6536|0.5846|
|PlayMusic| 0.5351| 0.4639|
|RateBook| 0.3651| 0.3515|
|SearchCreativeWork| 0.6922| 0.7956|
|SearchScreeningEvent|0.3354|0.3694|
|avg|0.5576|0.5031|