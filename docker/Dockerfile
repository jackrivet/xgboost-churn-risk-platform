FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1-cpu-py3

WORKDIR /opt/ml/processing/code

COPY processing/*.py /opt/ml/processing/code/

RUN pip install --no-cache-dir \
    pandas \
    pyarrow \
    scikit-learn \
    joblib
