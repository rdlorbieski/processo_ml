import algorithms.challenge_senai.predictor_train as pt
import repository.repository_service as rs


x_train, df_teste_treino, df_teste_real = pt.create_test_dataframes_to_train_and_validate()
if not rs.file_exists("models_to_train", "adaboost"):
    pt.create_model_to_predict_y_train(df_teste_treino)
print("O tamanho do teste utilizado para criar y_train é igual a ", df_teste_treino.shape[0], " e o tamanho da base de teste real é", df_teste_real.shape[0])
print(x_train.shape)

if not rs.dataset_train_exists():
    y_train = pt.predict_y_train(x_train)
    df_train = pt.concatXY(x_train, y_train, "target")
    rs.save_dataframe_train(df_train)

df_train = rs.get_dataset_train()
df_train = df_train.dropna(axis=0, how='any')


print(df_train.shape)
