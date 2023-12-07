

from deepcp.utils.registry import Registry

METRICS_REGISTRY = Registry("METRICS")


@METRICS_REGISTRY.register()
def coverage_rate(prediction_sets,labels):
    cvg  = 0
    for index,ele in enumerate(zip(prediction_sets,labels)):
        if ele[1] in ele[0]:
            cvg += 1
    return cvg/len(prediction_sets)

@METRICS_REGISTRY.register()
def average_size(prediction_sets,labels):
    avg_size = 0
    for index,ele in enumerate(prediction_sets):
        avg_size += len(ele)
    return avg_size/len(prediction_sets)




class Metrics:
    def __init__(self,metrics_list=[]) -> None:
        self.metrics_list = metrics_list
        
        
    def compute(self,prediction_sets,labels):
        metrics = {}
        for metric in self.metrics_list:
            metrics[metric] = METRICS_REGISTRY.get(metric)(prediction_sets,labels)
        return metrics
        
        
    