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

# 实验七

在RCSF的基础上，把query和context用bert分开编码，一条context对应一条query，然后concatenate起来通过一个self-attention

BookRestaurant: acc 0.4530, recall 0.3749, f1 0.4103

AddToPlaylist: acc 0.6912, recall 0.5320, f1 0.6012


# 实验八

在RCSF的基础上，把context和query在文本上concatenate(不加[sep])，输入bert，其余设置不变

BookRestaurant: Train epoch 1, step[4000 / 10116], Evalution acc=0.7418, recall=0.5085, f1=0.6034

BookRestaurant: Epoch 1, train_loss 0.052880, Evalution acc 0.6986, recall 0.4636, f1 0.5574

# 实验九

在实验6的基础上，用bert-large-uncased，dropout=0.2，bert后接两层self-attention

| 参数 | 参数值 |
| ---- | ----- |
|框架| torch |
|model|bert-large-uncased|
|warmup_rate|0.0|
|weight_decay|0.0|
|num_epochs|64|
|early_stop|64|
|n_top|5|
| batch_size | 4|
|context_max_len|64|
|label_max_len|32|
|dropout_rate|0.2|
|显卡|3090|

结果
|domain|baseline|my model|
|---|---|---|
|AddToPlaylist| 0.6870 |0.6244|
|BookRestaurant| 0.6349 |0.4289|
|GetWeather| 0.6536|0.5883|
|PlayMusic| 0.5351| 0.4896|
|RateBook| 0.3651| 0.3603|
|SearchCreativeWork| 0.6922| 0.6581|
|SearchScreeningEvent|0.3354|0.4446|
|avg|0.5576|0.5135|