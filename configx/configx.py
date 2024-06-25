class ConfigX(object):

    def __init__(self):
        super(ConfigX,self).__init__()
        self.approach = ['amalgam','bugLocator','blizzard']
        self.withoutFileList = []
        self.fileListsRow = []
        self.missRecFile = set()

        # filePath
        self.filepath_dict = {
            'zookeeper': {
                'bugInfo': '../data/bug_info/zookeeper_merged4tools.merged4tools',
                'bugReport': '../data/bug_report/zookeeper.xml',
                'bugFixedFileRanks': [
                    '../data/initial_recommendation_data/Amalgam/zookeeper_recommended',
                    '../data/initial_recommendation_data/bugLocator/zookeeper/bugFixedFileRanks',
                    '../data/initial_recommendation_data/blizzard/zookeeper/bugFixedFileRanks'

                ],
                'bugPredict': [
                    '../data/initial_recommendation_data/Amalgam/zookeeper_bugPredict.res.one4one',
                    '../data/initial_recommendation_data/bugLocator/zookeeper/zookeeper_bugPredict.res',
                    '../data/initial_recommendation_data/blizzard/zookeeper/zookeeper_bugPredict.res'
                ]
            },
            'openjpa': {
                'bugInfo': '../data/bug_info/openjpa_merged4tools.merged4tools',
                'bugReport': '../data/bug_report/openjpa.xml',
                'bugFixedFileRanks': [
                    '../data/initial_recommendation_data/Amalgam/openjpa_recommended',
                    '../data/initial_recommendation_data/bugLocator/openjpa/bugFixedFileRanks',
                    '../data/initial_recommendation_data/blizzard/openjpa/bugFixedFileRanks'

                ],
                'bugPredict': [
                    '../data/initial_recommendation_data/Amalgam/openjpa_bugPredict.res.one4one',
                    '../data/initial_recommendation_data/bugLocator/openjpa/openjpa_bugPredict.res',
                    '../data/initial_recommendation_data/blizzard/openjpa/openjpa_bugPredict.res'
                ]
            },
            'Tomcat': {
                'bugInfo': '../data/bug_info/tomcat_merged4tools',
                'bugReport': '../data/bug_report/tomcat.xml',
                'bugFixedFileRanks': [
                    '../data/initial_recommendation_data/Amalgam/tomcat_recommended',
                    '../data/initial_recommendation_data/bugLocator/tomcat/bugFixedFileRanks',
                    '../data/initial_recommendation_data/blizzard/tomcat/bugFixedFileRanks'

                ],
                'bugPredict': [
                    '../data/initial_recommendation_data/Amalgam/tomcat_bugPredict.res.one4one',
                    '../data/initial_recommendation_data/bugLocator/tomcat/tomcat_bugPredict.res',
                    '../data/initial_recommendation_data/blizzard/tomcat/tomcat_bugPredict.res'
                ]
            },
            'aspectj': {
                'bugInfo': '../data/bug_info/org.aspectj_merged4tools',
                'bugReport': '../data/bug_report/aspectj.xml',
                'bugFixedFileRanks': [
                    '../data/initial_recommendation_data/Amalgam/aspectj_recommended',
                    '../data/initial_recommendation_data/bugLocator/aspectj/bugFixedFileRanks',
                    '../data/initial_recommendation_data/blizzard/aspectj/bugFixedFileRanks'

                ],
                'bugPredict': [
                    '../data/initial_recommendation_data/Amalgam/org.aspectj_bugPredict.res.one4one',
                    '../data/initial_recommendation_data/bugLocator/aspectj/org.aspectj_bugPredict.res',
                    '../data/initial_recommendation_data/blizzard/aspectj/org.aspectj_bugPredict.res'
                ]
            },
            'hibernate': {
                'bugInfo': '../data/bug_info/hibernate-orm_merged4tools.merged4tools',
                'bugReport': '../data/bug_report/hibernate.xml',
                'bugFixedFileRanks': [
                    '../data/initial_recommendation_data/Amalgam/hibernate_recommended',
                    '../data/initial_recommendation_data/bugLocator/hibernate/bugFixedFileRanks',
                    '../data/initial_recommendation_data/blizzard/hibernate/bugFixedFileRanks'

                ],
                'bugPredict': [
                    '../data/initial_recommendation_data/Amalgam/hibernate-orm_bugPredict.res.one4one',
                    '../data/initial_recommendation_data/bugLocator/hibernate/hibernate-orm_bugPredict.res',
                    '../data/initial_recommendation_data/blizzard/hibernate/hibernate-orm_bugPredict.res'
                ]
            },
            'lucene': {
                'bugInfo': '../data/bug_info/lucene-solr_merged4tools.merged4tools',
                'bugReport': '../data/bug_report/lucene.xml',
                'bugFixedFileRanks': [
                    '../data/initial_recommendation_data/Amalgam/lucene_recommended',
                    '../data/initial_recommendation_data/bugLocator/lucene/bugFixedFileRanks',
                    '../data/initial_recommendation_data/blizzard/lucene/bugFixedFileRanks'

                ],
                'bugPredict': [
                    '../data/initial_recommendation_data/Amalgam/lucene-solr_bugPredict.res.one4one',
                    '../data/initial_recommendation_data/bugLocator/lucene/lucene-solr_bugPredict.res',
                    '../data/initial_recommendation_data/blizzard/lucene/lucene-solr_bugPredict.res'
                ]
            },
        }


        # train,test data
