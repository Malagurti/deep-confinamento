#Versão 2 do modelo

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

training_data = pd.read_csv("datasets/training_data_conf.csv", dtype = float)
test_data = pd.read_csv("datasets/test_data_conf.csv")

X_training = training_data.drop("total_vendido", axis= 1).values
Y_training = training_data[['total_vendido']].values

X_test = test_data.drop("total_vendido", axis=1).values
Y_test = test_data[['total_vendido']].values

X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

X_scaled_test = X_scaler.transform(X_test)
Y_scaled_test = Y_scaler.transform(Y_test)

#Hiperparametros
learning_rate = 0.001
num_epochs = 100
display_step = 5

#inputs e outputs

num_inputs = 6
num_outputs = 1

#Camadas
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

#Tensor board
RUN_NAME = "Execution with 50 100 50 nodes deploy"

#Camada de inputs com escopo de váriavel
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, num_inputs))

#Camada 1
with tf.variable_scope('layer1'):
    weights = tf.get_variable(name='weights1', shape=[num_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer= tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

#Camada 2
with tf.variable_scope('layer2'):
    weights = tf.get_variable(name='weights2', shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer= tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

#Camada 2
with tf.variable_scope('layer3'):
    weights = tf.get_variable(name='weights3', shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer= tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

#Camada Output
with tf.variable_scope('output'):
    weights = tf.get_variable(name='weights4', shape=[layer_3_nodes, num_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases4', shape=[num_outputs], initializer= tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

#Custo
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, num_outputs))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

#Otimizador
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Summary tensor borad

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    tf.summary.histogram('predict_value', prediction)
    summary = tf.summary.merge_all()


#Open Tensor flow

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    #Gravar resultado em diretorio
    training_writer = tf.summary.FileWriter("./logs/{}/training".format(RUN_NAME), session.graph)
    testing_writer = tf.summary.FileWriter("./logs/{}/testing".format(RUN_NAME), session.graph)

    for epoch in range(num_epochs):
        session.run(optimizer, feed_dict={X:X_scaled_training, Y: Y_scaled_training})

        #Progesso
        if epoch % display_step == 0:
            training_cost, training_summary = session.run([cost, summary],
                                                          feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            test_cost,test_summary = session.run([cost, summary], feed_dict={X: X_scaled_test, Y: Y_scaled_test})

            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(test_summary, epoch)

    print("Treinamento Concluido")

    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y:Y_scaled_training})
    final_test_cost = session.run(cost, feed_dict={X: X_scaled_test, Y:Y_scaled_test})

    print("Custo final do treinamento: {}".format(final_training_cost))
    print("Custo final do teste: {}".format(final_test_cost))

    Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_test})

    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

    total_vendido_real = test_data['total_vendido'].values[0]
    total_vendido_predicted = Y_predicted[0][0]

    print("\nTotal de Vendas real 1 lote: {}".format(total_vendido_real))
    print("Total de Vendas previstas de 1 lote: {}".format(total_vendido_predicted))

    #Salvando modelo em protobuff
    model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model")

    inputs = {'input': tf.saved_model.utils.build_tensor_info(X)}
    outputs = {'totalvendido': tf.saved_model.utils.build_tensor_info(prediction)}

    signature_def = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
                                                                           outputs=outputs,
                                                                           method_name=tf.saved_model.
                                                                           signature_constants.PREDICT_METHOD_NAME)

    model_builder.add_meta_graph_and_variables(
        session,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        }
    )

    model_builder.save()
