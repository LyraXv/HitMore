from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
# import Orange
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from itertools import combinations
from cliffs_delta import cliffs_delta

def compare_peformance(res_performance):
    t=pd.read_csv(res_performance,sep=",",header=0)
    t.columns=["Dataset","Technique","Acc@1","Acc@5","Acc@10","Acc@20","MAP","MRR",'fold']
    metric_list = ["Acc@1","Acc@5","Acc@10","Acc@20","MAP","MRR"]

    amalgam_list = ['Amalgam']
    bugLoctor_list = ['BugLocator']
    blizzard_list = ['Blizzard']

    # 将Acc@k列的百分数转换为小数
    acc_columns = ["Acc@1", "Acc@5", "Acc@10", "Acc@20"]
    t[acc_columns] = t[acc_columns].replace('%', '', regex=True).astype(float) / 100
    # t[metric_list] = t[metric_list].apply(pd.to_numeric, errors='coerce')
    t = t.sort_values(by=["Dataset", "fold"])
    print(t)


    for metric in metric_list:
        print(f"===Current Metric: {metric} ===")
        hitmore = t[metric][t['Technique']=='HitMore']
        amalgam = t[metric][t['Technique']=='Amalgam']
        buglocator = t[metric][t['Technique']=='BugLocator']
        blizzard = t[metric][t['Technique']=='Blizzard']

        # print(len(hitmore), len(amalgam), len(buglocator), len(blizzard))
        print(metric+'-rq1-ml-Friedman-test:',friedmanchisquare(hitmore, amalgam, buglocator, blizzard))
        data = np.array([hitmore, amalgam, buglocator, blizzard])
        #print(sp.posthoc_nemenyi_friedman(data.T))

        #pairwise comparision between ML algorithms
        comp_list=['HitMore', 'Amalgam', 'BugLocator', 'Blizzard']
        combs = np.fromiter(combinations(range(len(comp_list)),2), dtype='i,i')

        for (i,j) in combs:
            idat=t[metric][t["Technique"]==comp_list[i]]
            jdat=t[metric][t["Technique"]==comp_list[j]]

            d, res = cliffs_delta(idat, jdat)
            print(comp_list[i],comp_list[j],d,res)

            idat = pd.to_numeric(idat, errors='coerce')
            jdat = pd.to_numeric(jdat, errors='coerce')
            idat = np.array(idat, dtype=float)
            jdat = np.array(jdat, dtype=float)

            res_str = '%.3f'%d + '('+res+')***'
            p_value = wilcoxon(idat,jdat)

            if  comp_list[i]=='HitMore' and comp_list[j]=='Amalgam':
                amalgam_list.append(res_str)
                print(wilcoxon(idat, jdat))
            if  comp_list[i]=='HitMore' and comp_list[j]=='BugLocator':
                bugLoctor_list.append(res_str)
                print(wilcoxon(idat,jdat))
            if  comp_list[i]=='HitMore' and comp_list[j]=='Blizzard':
                blizzard_list.append(res_str)
                print(wilcoxon(idat,jdat))
    res_effect_size = pd.DataFrame([amalgam_list,bugLoctor_list,blizzard_list],columns=['Technique',"Acc@1","Acc@5","Acc@10","Acc@20","MAP","MRR"])
    res_effect_size.to_csv("../data/results/hitmore-performance-effect-size.csv")

def compare_effectiveness(res_effectiveness):
    t=pd.read_csv(res_effectiveness,sep=",",header=0)
    t.columns=["Dataset","Technique","HitCount_1","HitCount_2","HitCount_all","HitCount_mult","Multi_num","mC",'fold']
    metric_list = ["HitCount_1","HitCount_2","HitCount_all","mC"]

    amalgam_list = ['Amalgam']
    bugLoctor_list = ['BugLocator']
    blizzard_list = ['Blizzard']

    t = t.sort_values(by=["Dataset", "fold"])
    print(t)


    for metric in metric_list:
        print(f"===Current Metric: {metric} ===")
        hitmore = t[metric][t['Technique']=='HitMore']
        amalgam = t[metric][t['Technique']=='Amalgam']
        buglocator = t[metric][t['Technique']=='BugLocator']
        blizzard = t[metric][t['Technique']=='Blizzard']

        # print(len(hitmore), len(amalgam), len(buglocator), len(blizzard))
        print(metric+'-rq1-ml-Friedman-test:',friedmanchisquare(hitmore, amalgam, buglocator, blizzard))
        data = np.array([hitmore, amalgam, buglocator, blizzard])
        #print(sp.posthoc_nemenyi_friedman(data.T))

        #pairwise comparision between ML algorithms
        comp_list=['HitMore', 'Amalgam', 'BugLocator', 'Blizzard']
        combs = np.fromiter(combinations(range(len(comp_list)),2), dtype='i,i')

        for (i,j) in combs:
            idat=t[metric][t["Technique"]==comp_list[i]]
            jdat=t[metric][t["Technique"]==comp_list[j]]

            d, res = cliffs_delta(idat, jdat)
            print(comp_list[i],comp_list[j],d,res)

            idat = pd.to_numeric(idat, errors='coerce')
            jdat = pd.to_numeric(jdat, errors='coerce')
            idat = np.array(idat, dtype=float)
            jdat = np.array(jdat, dtype=float)

            res_str = '%.3f'%d + '('+res+')*'
            p_value = wilcoxon(idat,jdat)

            if  comp_list[i]=='HitMore' and comp_list[j]=='Amalgam':
                amalgam_list.append(res_str)
                print(wilcoxon(idat, jdat))
            if  comp_list[i]=='HitMore' and comp_list[j]=='BugLocator':
                bugLoctor_list.append(res_str)
                print(wilcoxon(idat,jdat))
            if  comp_list[i]=='HitMore' and comp_list[j]=='Blizzard':
                blizzard_list.append(res_str)
                print(wilcoxon(idat,jdat))
    res_effect_size = pd.DataFrame([amalgam_list,bugLoctor_list,blizzard_list],columns=['Technique',"HitCount_1","HitCount_2","HitCount_all","mC"])
    res_effect_size.to_csv("../data/results/hitmore-effectiveness-effect-size-1.csv")


if __name__ == "__main__":
    # res_performance = f"../data/results/HitMore_performance_folds.csv"
    # compare_peformance(res_performance)
    # res_effectiveness = f"../data/results/HitMore_effectiveness_folds.csv"
    res_effectiveness = f"../data/results/effectiveness_test.csv"
    compare_effectiveness(res_effectiveness)