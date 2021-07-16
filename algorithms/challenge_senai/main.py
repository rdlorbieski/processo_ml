import algorithms.challenge_senai.predictor_train as pt
import repository.repository_service as rs
import algorithms.challenge_senai.predictor_test as pte

x_train, df_teste_treino, df_teste_real = pt.create_test_dataframes_to_train_and_validate()
tree_model = pt.create_model_to_predict_y_train(df_teste_treino)
if not rs.file_exists("models_to_train", "tree"):
    rs.save_model_to_train(tree_model, "tree")

print("O tamanho do teste utilizado para criar y_train é igual a ", df_teste_treino.shape[0], " e o tamanho da base de teste real é", df_teste_real.shape[0])
print(x_train.shape)

if not rs.dataset_trainer_test_exists():
    rs.save_dataframe_test("df_trainer_test", df_teste_treino)

if not rs.dataset_test_validation_exists():
    rs.save_dataframe_test("df_test_validation", df_teste_real)

if not rs.dataset_train_exists():
    y_train = pt.predict_y_train(x_train)
    df_train = pt.concatXY(x_train, y_train, "target")
    df_train = df_train.dropna(axis=0, how='any')
    rs.save_dataframe_train(df_train)

# ======================================= DF_TRAIN READY, START EVALUATING SOLUTION:
df_train = rs.get_dataset_train()
df_train = df_train.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14','c16', 'c18', 'c19', 'target']]
df_train_clean = pt.remove_outliers(0.2, df_train)
df_train_clean_normalized = pte.normalize_dataframe(df_train_clean)


df_test_real = rs.get_dataset_test("df_test_validation")
df_test_real = df_test_real.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14','c16', 'c18', 'c19', 'target']]
df_test_real_normalized = pte.normalize_dataframe(df_test_real)

pte.create_and_save_models(df_train_clean_normalized, df_test_real_normalized)









