

def compute_coverage_rate(prediction_sets,labels):
    cvg  = 0
    for index,ele in enumerate(zip(prediction_sets,labels)):
        if ele[1] in ele[0]:
            cvg += 1
    return cvg/len(prediction_sets)


class Metrics:
    def __init__(self,metrics_list=[]) -> None:
        self.metrics_list = metrics_list
        
        
    def compute(self,prediction_sets,labels):
        # for metric in self.metrics_list:
        return compute_coverage_rate(prediction_sets,labels)
        
        
    