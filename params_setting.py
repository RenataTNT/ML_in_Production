# build dataset params #
def build_dataset_params():
    CHURNED_START_DATE = '2019-09-01'
    CHURNED_END_DATE = '2019-10-01'

    INTER_1 = (1,7)
    INTER_2 = (8,14)
    INTER_3 = (15,21)
    INTER_4 = (22,28)

    INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]
    return CHURNED_START_DATE, CHURNED_END_DATE, INTER_LIST

#