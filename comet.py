# # import comet_ml at the top of your file
# import time
#
# from comet_ml import ExistingExperiment, Experiment
# from comet_ml.api import API
#
#
# class MyExperiment(object):
#     _instance = None
#     experiment = None  # type: Experiment
#
#     def __new__(cls, *args, **kw):
#         if cls._instance is None:
#             cls._instance = object.__new__(cls)
#             proj_name = args[0]
#             exp_name = args[1]
#             api_key = "ebB20srr60aW04Y5rPJIaH0zV"
#             comet_api = API(api_key=api_key)
#             cls._instance.experiment = comet_api.get(f"yingtaomj/{proj_name}/{exp_name}")
#             if cls._instance.experiment is None:
#                 cls._instance.experiment = Experiment(
#                     api_key=api_key,
#                     project_name=proj_name,
#                     workspace="yingtaomj",
#                     log_code=False,
#                     auto_output_logging=False,
#                 )
#                 cls._instance.experiment.set_name(exp_name)
#             else:
#                 cls._instance.experiment = ExistingExperiment(api_key=api_key,
#                                                               previous_experiment=cls._instance.experiment.id,
#                                                               log_code=False, auto_output_logging=False,
#                                                               )
#         return cls._instance
#
#     def __init__(self, proj_name, exp_name):
#         pass
#
#
#
# if __name__ == '__main__':
#     exp = MyExperiment('general', 'll2')
#     exp = MyExperiment('general', 'll2')
#     for step in range(30, 50):
#         exp.experiment.log_other(key='lala', value=step)
#         print(step)
#         time.sleep(10)
