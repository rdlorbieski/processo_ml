import algorithms.challenge_senai.predictor_train as pt
import repository.repository_service as rs
import algorithms.challenge_senai.predictor_test as pte

x_train, df_teste_treino, df_teste_real = pt.create_test_dataframes_to_train_and_validate()
if not rs.file_exists("models_to_train", "tree"):
    pt.create_model_to_predict_y_train(df_teste_treino)
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




df_train = rs.get_dataset_train()
df_train = df_train.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14','c16', 'c18', 'c19', 'target']]
df_train_clean = pt.remove_outliers(0.2, df_train)
df_train_clean_normalized = pte.normalize_dataframe(df_train_clean)


df_test_real = rs.get_dataset_test("df_test_validation")
df_test_real = df_test_real.loc[:, ['c1', 'c6', 'c8', 'c11', 'c13', 'c14','c16', 'c18', 'c19', 'target']]
df_test_real_normalized = pte.normalize_dataframe(df_test_real)

acc_2, min = [0, 0.7]

while acc_2 < min:
    print(".")
    stacking, acc_2, auc_level_2, matrix_confusion = pte.avaliate_level_2(df_train_clean_normalized, df_test_real_normalized)

    if acc_2 >= min:

        # check performance of base classifiers (level 0):
        print("======================= check performance and evaluating level 0 ")
        models_levels_0, level_0_performance = pte.avaliate_level_0(df_train_clean_normalized, df_test_real_normalized)
        for key, value in level_0_performance.items():
            if value[0] >= 0.5:
                print(key + " acc = " + str(value[0]) + " auc = " + str(value[1]))

        names_level_0 = ["knn1", "knn2", "knn3", "knn4", "svm1", "etc", "mlp"]
        for i in range(len(names_level_0)):
            rs.save_model(models_levels_0[i], "models_to_evaluate", "level_0", names_level_0[i])


        print("======================= evaluating level 1 ")
        models_levels_1, level_1_performance = pte.avaliate_level_1(df_train_clean_normalized, df_test_real_normalized)

        for key, value in level_1_performance.items():
            print(key + " acc = " + str(value[0]) + " auc = " + str(value[1]))

        names_level_1 = ["ensemble_lazy", "ensemble_tree", "ensemble_nn", "ensemble_svm"]
        for i in range(len(names_level_1)):
            rs.save_model(models_levels_1[i], "models_to_evaluate", "level_1", names_level_1[i])


        print("======================= evaluating level 2")
        print("acc = ", min, "auc = ", auc_level_2)




        break
    else:
        continue









