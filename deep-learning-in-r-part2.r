{
 "cells": [
  {
   "cell_type": "raw",
   "id": "cd8fa8b3",
   "metadata": {
    "papermill": {
     "duration": 0.004763,
     "end_time": "2023-05-29T16:21:11.101625",
     "exception": false,
     "start_time": "2023-05-29T16:21:11.096862",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Load required libraries\n",
    "library(keras)\n",
    "library(tensorflow)\n",
    "library(ggplot2)\n",
    "\n",
    "# Load MNIST dataset\n",
    "mnist <- dataset_mnist()\n",
    "x_train <- mnist$train$x\n",
    "y_train <- mnist$train$y\n",
    "x_test <- mnist$test$x\n",
    "y_test <- mnist$test$y\n",
    "\n",
    "# Preprocess the data\n",
    "x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))\n",
    "x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))\n",
    "x_train <- x_train / 255\n",
    "x_test <- x_test / 255\n",
    "\n",
    "# Define the model architecture\n",
    "model <- keras_model_sequential()\n",
    "model %>%\n",
    "  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = \"relu\", input_shape = c(28, 28, 1)) %>%\n",
    "  layer_max_pooling_2d(pool_size = c(2, 2)) %>%\n",
    "  layer_dropout(0.25) %>%\n",
    "  layer_flatten() %>%\n",
    "  layer_dense(units = 128, activation = \"relu\") %>%\n",
    "  layer_dropout(0.5) %>%\n",
    "  layer_dense(units = 10, activation = \"softmax\")\n",
    "\n",
    "# Compile the model\n",
    "model %>% compile(\n",
    "  loss = \"sparse_categorical_crossentropy\",\n",
    "  optimizer = optimizer_adam(),\n",
    "  metrics = c(\"accuracy\")\n",
    ")\n",
    "\n",
    "# Train the model\n",
    "history <- model %>% fit(\n",
    "  x_train, y_train,\n",
    "  epochs = 10,\n",
    "  batch_size = 128,\n",
    "  validation_split = 0.2\n",
    ")\n",
    "\n",
    "# Evaluate the model\n",
    "accuracy <- model %>% evaluate(x_test, y_test)[[2]]\n",
    "cat(\"Accuracy: \", accuracy, \"\\n\")\n",
    "\n",
    "# Plot accuracy and loss curves\n",
    "plot(history)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1f0f230",
   "metadata": {
    "papermill": {
     "duration": 0.003363,
     "end_time": "2023-05-29T16:21:11.108907",
     "exception": false,
     "start_time": "2023-05-29T16:21:11.105544",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Loading Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d105ca52",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:21:11.120588Z",
     "iopub.status.busy": "2023-05-29T16:21:11.118179Z",
     "iopub.status.idle": "2023-05-29T16:21:12.519265Z",
     "shell.execute_reply": "2023-05-29T16:21:12.517611Z"
    },
    "papermill": {
     "duration": 1.409914,
     "end_time": "2023-05-29T16:21:12.522270",
     "exception": false,
     "start_time": "2023-05-29T16:21:11.112356",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load required libraries\n",
    "library(keras)\n",
    "library(tensorflow)\n",
    "library(ggplot2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9004aa9c",
   "metadata": {
    "papermill": {
     "duration": 0.003336,
     "end_time": "2023-05-29T16:21:12.529112",
     "exception": false,
     "start_time": "2023-05-29T16:21:12.525776",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Loading the Data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "101aac80",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:21:12.586997Z",
     "iopub.status.busy": "2023-05-29T16:21:12.537528Z",
     "iopub.status.idle": "2023-05-29T16:21:24.042482Z",
     "shell.execute_reply": "2023-05-29T16:21:24.040794Z"
    },
    "papermill": {
     "duration": 11.513145,
     "end_time": "2023-05-29T16:21:24.045634",
     "exception": false,
     "start_time": "2023-05-29T16:21:12.532489",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Load MNIST dataset\n",
    "mnist <- dataset_mnist()\n",
    "x_train <- mnist$train$x\n",
    "y_train <- mnist$train$y\n",
    "x_test <- mnist$test$x\n",
    "y_test <- mnist$test$y\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e8a3a7e",
   "metadata": {
    "papermill": {
     "duration": 0.003362,
     "end_time": "2023-05-29T16:21:24.052758",
     "exception": false,
     "start_time": "2023-05-29T16:21:24.049396",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c8a90182",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:21:24.062687Z",
     "iopub.status.busy": "2023-05-29T16:21:24.061272Z",
     "iopub.status.idle": "2023-05-29T16:21:25.548405Z",
     "shell.execute_reply": "2023-05-29T16:21:25.546712Z"
    },
    "papermill": {
     "duration": 1.495108,
     "end_time": "2023-05-29T16:21:25.551215",
     "exception": false,
     "start_time": "2023-05-29T16:21:24.056107",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Preprocess the data\n",
    "x_train <- array_reshape(x_train, c(dim(x_train)[1], 28, 28, 1))\n",
    "x_test <- array_reshape(x_test, c(dim(x_test)[1], 28, 28, 1))\n",
    "x_train <- x_train / 255\n",
    "x_test <- x_test / 255"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "055dccab",
   "metadata": {
    "papermill": {
     "duration": 0.003329,
     "end_time": "2023-05-29T16:21:25.558028",
     "exception": false,
     "start_time": "2023-05-29T16:21:25.554699",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Defining model Architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d6c51e66",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:21:25.567684Z",
     "iopub.status.busy": "2023-05-29T16:21:25.566433Z",
     "iopub.status.idle": "2023-05-29T16:21:28.760006Z",
     "shell.execute_reply": "2023-05-29T16:21:28.758166Z"
    },
    "papermill": {
     "duration": 3.201557,
     "end_time": "2023-05-29T16:21:28.763134",
     "exception": false,
     "start_time": "2023-05-29T16:21:25.561577",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Define the model architecture\n",
    "model <- keras_model_sequential()\n",
    "model %>%\n",
    "  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = \"relu\", input_shape = c(28, 28, 1)) %>%\n",
    "  layer_max_pooling_2d(pool_size = c(2, 2)) %>%\n",
    "  layer_dropout(0.25) %>%\n",
    "  layer_flatten() %>%\n",
    "  layer_dense(units = 128, activation = \"relu\") %>%\n",
    "  layer_dropout(0.5) %>%\n",
    "  layer_dense(units = 10, activation = \"softmax\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b9c8684",
   "metadata": {
    "papermill": {
     "duration": 0.003641,
     "end_time": "2023-05-29T16:21:28.770549",
     "exception": false,
     "start_time": "2023-05-29T16:21:28.766908",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Compiling the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a6ee8b7e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:21:28.780191Z",
     "iopub.status.busy": "2023-05-29T16:21:28.778822Z",
     "iopub.status.idle": "2023-05-29T16:21:28.807787Z",
     "shell.execute_reply": "2023-05-29T16:21:28.806244Z"
    },
    "papermill": {
     "duration": 0.036457,
     "end_time": "2023-05-29T16:21:28.810403",
     "exception": false,
     "start_time": "2023-05-29T16:21:28.773946",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Compile the model\n",
    "optimizer <- tf$keras$optimizers$Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = FALSE)\n",
    "model %>% compile(\n",
    "  loss = \"sparse_categorical_crossentropy\",\n",
    "  optimizer = optimizer,\n",
    "  metrics = c(\"accuracy\")\n",
    ")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "39ccd4ef",
   "metadata": {
    "papermill": {
     "duration": 0.003369,
     "end_time": "2023-05-29T16:21:28.818480",
     "exception": false,
     "start_time": "2023-05-29T16:21:28.815111",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Training the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "17b33235",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-05-29T16:21:28.828679Z",
     "iopub.status.busy": "2023-05-29T16:21:28.827336Z",
     "iopub.status.idle": "2023-05-29T16:21:55.095319Z",
     "shell.execute_reply": "2023-05-29T16:21:55.093462Z"
    },
    "papermill": {
     "duration": 26.276533,
     "end_time": "2023-05-29T16:21:55.099020",
     "exception": false,
     "start_time": "2023-05-29T16:21:28.822487",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "# Train the model\n",
    "history <- model %>% fit(\n",
    "  x_train, y_train,\n",
    "  epochs = 10,\n",
    "  batch_size = 128,\n",
    "  validation_split = 0.2\n",
    ")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "R",
   "language": "R",
   "name": "ir"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "4.0.5"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 48.182954,
   "end_time": "2023-05-29T16:21:56.327665",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-05-29T16:21:08.144711",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
