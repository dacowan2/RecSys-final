import pandas as pd
model_info_all_list = []
for model_num in range(72):
    model_info_df = pd.read_csv(f'models/model_{model_num}/model_info.csv')
    model_info_single_list = list(model_info_df.iloc[0])[1:]
    model_info_all_list.append(model_info_single_list)

model_info_header_list = ['model', 'test mse', 'test preds std', 'epochs', 'learning rate', 'n_nodes_per_layer', 'n_factors', 'batch_size', 'dropout_prob']
model_info_all_df = pd.DataFrame.from_records(model_info_all_list, columns = model_info_header_list)

model_info_all_df.to_csv('models/model_info-all.csv')