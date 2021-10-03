# %%設定追蹤URI
import mlflow
mlflow.set_registry_uri("http://192.168.1.3:5000")
mr_uri = mlflow.get_registry_uri()
print("Current registry uri: {}".format(mr_uri))

print(mlflow.tracking.is_tracking_uri_set())
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))
# %%獲取實驗資訊
experiment = mlflow.get_experiment(experiment_id="0")
# experiment = mlflow.get_experiment_by_name(name = "Select architecture")
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
# %% 選擇一個實驗(與create_experiment()不同)
# Set an experiment name, which must be unique and case sensitive.
# 如果指定的實驗不存在，則會生出一個新的實驗。
mlflow.set_experiment("Homework")

# Get Experiment Details
experiment = mlflow.get_experiment_by_name("Homework")
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
# %%開關run

# Start run and get status
mlflow.start_run(run_name="run for HW")
run = mlflow.active_run()
print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))

# End run and get status
mlflow.end_run()
run = mlflow.get_run(run.info.run_id)  # 可惜只能從run id去get，因為run name不是唯一值
print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
print("--")

# Check for any active runs
print("Active run: {}".format(mlflow.active_run()))

# 取得run資訊
run_id = "8fc7beafa3164d37afb2ce263e13cc6b"
print("run_id: {}; lifecycle_stage: {}".format(run_id,
                                               mlflow.get_run(run_id).info.lifecycle_stage))
# %% 檢查URL
registry_uri = mlflow.get_registry_uri()
print("Current registry uri: {}".format(registry_uri))
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))


# %%

# Create nested runs
with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
        mlflow.log_param("child", "yes")

print("parent run_id: {}".format(parent_run.info.run_id))
print("child run_id : {}".format(child_run.info.run_id))
print("--")

# Search all child runs with a parent id
query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
results = mlflow.search_runs(filter_string=query)
print(results[["run_id", "params.child", "tags.mlflow.runName"]])
