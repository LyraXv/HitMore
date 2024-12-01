import math

import numpy as np
import pandas as pd

from configx.configx import ConfigX
from utils_rdFiles import readRecList, search_bugCmit, openCodeCorpus, searchAllContent
from utils_readability import get_words, get_char_count, get_sentences, count_syllables, count_complex_words


class Readability:
    analyzedVars = {}

    def __init__(self, text):
        self.analyze_text(text)

    def analyze_text(self, text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        words = get_words(text)
        char_count = get_char_count(words)

        word_count = len(words)
        sentence_count = len(get_sentences(text))
        syllable_count = count_syllables(words)
        complexwords_count = count_complex_words(text)
        avg_words_p_sentence = word_count / sentence_count

        self.analyzedVars = {
            'words': words,
            'char_cnt': float(char_count),
            'word_cnt': float(word_count),
            'sentence_cnt': float(sentence_count),
            'syllable_cnt': float(syllable_count),
            'complex_word_cnt': float(complexwords_count),
            'avg_words_p_sentence': float(avg_words_p_sentence)
        }

    def ARI(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 4.71 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt']) + 0.5 * (
                        self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt']) - 21.43
        return score

    def FleschReadingEase(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 206.835 - (1.015 * (self.analyzedVars['avg_words_p_sentence'])) - (
                        84.6 * (self.analyzedVars['syllable_cnt'] / self.analyzedVars['word_cnt']))
        return round(score, 4)

    def FleschKincaidGradeLevel(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.39 * (self.analyzedVars['avg_words_p_sentence']) + 11.8 * (
                        self.analyzedVars['syllable_cnt'] / self.analyzedVars['word_cnt']) - 15.59
        return round(score, 4)

    def GunningFogIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.4 * ((self.analyzedVars['avg_words_p_sentence']) + (
                        100 * (self.analyzedVars['complex_word_cnt'] / self.analyzedVars['word_cnt'])))
        return round(score, 4)

    def SMOGIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (math.sqrt(self.analyzedVars['complex_word_cnt'] * (30 / self.analyzedVars['sentence_cnt'])) + 3)
        return score

    def ColemanLiauIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (5.89 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt'])) - (
                        30 * (self.analyzedVars['sentence_cnt'] / self.analyzedVars['word_cnt'])) - 15.8
        return round(score, 4)

    def LIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt'] + float(100 * longwords) / \
                    self.analyzedVars['word_cnt']
        return score

    def RIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = longwords / self.analyzedVars['sentence_cnt']
        return score


def get_code_readability(rec_lists):
    grouped = rec_lists.groupby('bugId')
    res_list = [] # output
    for bugId,group in grouped:
        # print("BUGID: ",bugId)
        CodeCorpus_None=0
        bugCmit = search_bugCmit(bugId,dataset)
        print bugCmit
        CodeCorpus = openCodeCorpus(dataset,bugCmit)
        for index, file in group.iterrows():
            res = []
            filepaths ={'path_0':file['path_0'],
                        'path_1':file['path_1'],
                        'path_2':file['path_2']}
            all_contens = searchAllContent(CodeCorpus,filepaths)
            if all_contens is not None:
                rd = Readability(all_contens)
                ARI = rd.ARI()
                FleschReadingEase = rd.FleschReadingEase()
                FleschKincaidGradeLevel= rd.FleschKincaidGradeLevel()
                GunningFogIndex = rd.GunningFogIndex()
                SMOGIndex = rd.SMOGIndex()
                ColemanLiauIndex = rd.ColemanLiauIndex()
                LIX = rd.LIX()
                RIX = rd.RIX()
            else:
                ARI = np.NaN
                FleschReadingEase = np.NaN
                FleschKincaidGradeLevel= np.NaN
                GunningFogIndex = np.NaN
                SMOGIndex = np.NaN
                ColemanLiauIndex = np.NaN
                LIX = np.NaN
                RIX = np.NaN
                CodeCorpus_None +=1

            res.append(index)
            res.append(bugId)
            # readability features:
            res.append(ARI)
            res.append(FleschReadingEase)
            res.append(FleschKincaidGradeLevel)
            res.append(GunningFogIndex)
            res.append(SMOGIndex)
            res.append(ColemanLiauIndex)
            res.append(LIX)
            res.append(RIX)
            # pack
            res_list.append(res)
        if CodeCorpus_None != 0:
            print("=====BugId:",bugId,"CodeCorpus_None:",CodeCorpus_None)
    df_readability = pd.DataFrame(res_list,columns=['index','bugId',
                                                    'rd.ARI','rd.FleschReadingEase','rd.FleschKincaidGradeLevel',
                                                    'rd.GunningFogIndex','rd.SMOGIndex','rd.ColemanLiauIndex',
                                                    'rd.LIX','rd.RIX'])
    save_path = "../data/splited_and_boosted_data/"+dataset+"/buggyFileFeatures/CodeReadability/"+str(i)+".csv"
    df_readability.to_csv(save_path,index=False)

if __name__ == "__main__":
    # CodeCorpus
    configx = ConfigX()
    for dataset, file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']: #zoo openjpa tomcat aspectj hibernate
            continue
        print("=====Code Readability: ",dataset)
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print("Current fold: ",i)
            rec_lists = readRecList(dataset, i)
            get_code_readability(rec_lists)