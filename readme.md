# 实验一
参数
加上了ent_embedding和pos_embedding, label knowledge和utterance之间用点积注意力
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
在实验一的基础上去掉了ent_embedding和pos_embedding，其余参数没变，label knowledge和utterance之间用点积注意力
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

# 实验四
RCSF模型，用自己的代码实现

AddToPlaylist RateBook PlayMusic BookRestaurant这四个domain的参数：
| 参数 | 参数值 |
| ---- | ----- |
|框架| torch |
|model|bert-large-uncased|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|15|
|n_top|5|
| batch_size | 32|
|max_len|128|
|显卡|3090|
SearchScreeningEvent GetWeather SearchCreativeWork这三个domain的参数：

| 参数 | 参数值 |
| ---- | ----- |
|框架| torch |
|model|bert-large-uncased|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|15|
|n_top|5|
| batch_size | 8|
|max_len|128|
|显卡|3090|

结果
|domain|baseline|baseline我的实现|
|---|---|---|
|AddToPlaylist| 0.6870 |0.6676|
|BookRestaurant| 0.6349 |0.6260|
|GetWeather| 0.6536|0.6948|
|PlayMusic| 0.5351| 0.5574|
|RateBook| 0.3651| 0.4094|
|SearchCreativeWork| 0.6922| 0.6593|
|SearchScreeningEvent|0.3354|0.3728|
|avg|0.5576|0.5696|

从实验结果来看，early_stop设成大于10比较好

# 实验五

在实验2的基础上，把bert-large换成debertav3-large，将label knowledge和utterance分别编码，然后concatenate后输入multi-head self-attention层

| 参数 | 参数值 |
| ---- | ----- |
|框架| torch |
|model|debertav3-large|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|64|
|n_top|5|
| batch_size | 8|
|context_max_len|64|
|label_max_len|32|
|显卡|3090|

结果
|domain|baseline|my model|
|---|---|---|
|AddToPlaylist| 0.6870 |0.7173|
|BookRestaurant| 0.6349 |0.4349|
|GetWeather| 0.6536|0.6263|
|PlayMusic| 0.5351| 0.4544|
|RateBook| 0.3651| 0.3656|
|SearchCreativeWork| 0.6922| 0.8104|
|SearchScreeningEvent|0.3354|0.5129|
|avg|0.5576|0.5603|


# 实验六

在实验5的基础上，把debertav3-large换成了bert-large-uncased

| 参数 | 参数值 |
| ---- | ----- |
|框架| torch |
|model|bert-large-uncased|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|64|
|n_top|5|
| batch_size | 8|
|context_max_len|64|
|label_max_len|32|
|显卡|3090|

结果
|domain|baseline|my model|
|---|---|---|
|AddToPlaylist| 0.6870 |0.6892|
|BookRestaurant| 0.6349 |0.4248|
|GetWeather| 0.6536|0.6465|
|PlayMusic| 0.5351| 0.4274|
|RateBook| 0.3651| 0.3390|
|SearchCreativeWork| 0.6922| 0.7765|
|SearchScreeningEvent|0.3354|0.4305|
|avg|0.5576|0.5330|