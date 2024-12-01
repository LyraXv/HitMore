class ConfigX(object):

    def __init__(self):
        super(ConfigX,self).__init__()

        self.approach = ['amalgam','bugLocator','blizzard']
        self.RecModel = 'XGboost'  #XGboost/Adaboost
        self.rawDataType = 'Zou'
        self.preprocess = True
        self.withoutFileList = []
        self.fileListsRow = []
        self.missRecFile = set()
        self.saveTop20RecLists = False
        self.saveInitialRecLists = True


        # filePath
        self.filepath_dict = {
            'zookeeper': {
                'bugInfo': '../data/bug_info/zookeeper_merged4tools.merged4tools',
                'bugReport': '../data/bug_report/zookeeper.xml',
                'bugReports': '../data/BugReports/zookeeper.xml',
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
                'bugReports': '../data/BugReports/openjpa.xml',
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
                'bugReports': '../data/BugReports/tomcat.xml',
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
                'bugReports': '../data/BugReports/org.aspectj.xml',
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
                'bugReports': '../data/BugReports/hibernate-orm.xml',
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
                'bugReports': '../data/BugReports/lucene-solr.xml',
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
        # keywords
        self.keywords={
            'zookeeper':['caus', 'zookeep', 'client', 'would', 'configur', 'creat', 'sasl', 'use', 'happen', 'zk', 'tri', 'version', 'server', 'make', 'current', 'case', 'build', 'process', 'file', 'add', 'also', 'look', 'one', 'need', 'port', 'request', 'test', 'get', 'check', 'data', 'node', 'cluster', 'set', 'leader', 'zxid','new', 'shut', 'attempt', 'connect', 'socket', 'timeout', 'open', 'quorum', 'fail', 'elect', 'start', 'notif', 'send', 'default', 'time', 'messag', 'follow', 'receiv', 'accept', 'wait', 'issu', 'like', 'fix', 'run', 'code', 'error', 'debug', 'found', 'problem', 'session', 'state', 'snapshot', 'path', 'chang', 'thread', 'see','address', 'authent', 'null', 'return', 'call', 'log', 'myid', 'info', 'method', 'except', 'complet', 'command', 'read', 'close', 'shutdown', 'stat', 'gt', 'c', 'warn', 'java', 'exit', 'id', 'establish', 'myid', 'nleader', 'nzxid', 'nround', 'nstate', 'nsid', 'npeerepoch', 'myid', 'worker',],
            'openjpa':['support', 'string', 'version', 'need', 'test', 'time', 'field', 'creat', 'tabl', 'one', 'use', 'like', 'user', 'databas', 'gt', 'see', 'updat', 'code', 'annot', 'type', 'map', 'fail', 'run', 'set', 'call', 'implement', 'caus', 'jpa', 'order', 'would', 'openjpa', 'provid', 'follow', 'except', 'data', 'result', 'column', 'sourc', 'int', 'tri', 'work', 'null', 'return', 'chang', 'class', 'persist', 'problem', 'issu', 'properti', 'entiti', 'configur', 'file', 'applic', 'name', 'default', 'join', 'queri', 'contain', 'execut', 'list', 'id', 'error', 'privat', 'new', 'public', 'long', 'select', 'case', 'method', 'errorgt', 'trace', 'info', 'paramet', 'enhanc', 'object', 'key', 'get', 'valu', 'gener', 'sql', 'instanc', 'cach', 'load','ltproperti', 'main', 'java'],
            'Tomcat':['work', 'attribut', 'follow', 'file', 'tomcat', 'fail', 'except', 'line', 'contain', 'use', 'problem','configur', 'set', 'call', 'sourc', 'see', 'one', 'method', 'like', 'tri', 'code', 'error', 'debug', 'class', 'messag', 'would', 'chang', 'thread', 'applic', 'get', 'user', 'result', 'case', 'new', 'request', 'exampl', 'http', 'server', 'creat', 'return', 'null', 'also', 'respons', 'test', 'name', 'connector', 'access', 'load', 'directori', 'valu', 'ad', 'path', 'public', 'string', 'java', 'version', 'instal', 'start', 'run', 'fix', 'throw', 'log', 'bug', 'attach', 'url', 'includ', 'caus', 'found', 'gener',  'default', 'page', 'apach', 'time', 'servlet', 'webapp', 'deploy', 'context', 'process', 'connect', 'object', 'paramet', 'web', 'jsp', 'info', 'servic', 'seem', 'session', 'tag', 'compil'],
            'aspectj':['ltw', 'return', 'type', 'gener', 'except', 'compil', 'messag', 'declar', 'warn', 'error', 'field', 'build', 'annot', 'method', 'use', 'weav', 'class', 'aspectj', 'itd', 'fail', 'java', 'caus', 'around', 'aspect', 'interfac', 'support', 'pointcut', 'call', 'file', 'match', 'advic', 'doesnt', 'increment'],
            'hibernate':['fail', 'column', 'gener', 'default', 'method', 'creat', 'tabl', 'hibern', 'databas', 'use', 'queri', 'select', 'class', 'work', 'type', 'need', 'insert', 'entiti', 'contain', 'map', 'public', 'id', 'privat', 'case', 'sql', 'valu', 'get', 'properti', 'paramet', 'annot', 'one', 'user', 'set', 'issu', 'new', 'could', 'implement', 'close', 'like', 'order', 'follow', 'except', 'caus', 'line', 'problem','session', 'result', 'see', 'updat', 'name', 'long', 'version', 'string', 'key', 'also', 'null', 'cach', 'exampl', 'file', 'list', 'field', 'add', 'test', 'error', 'fals', 'chang', 'gt', 'execut', 'would', 'code', 'join', 'return', 'load', 'support', 'persist', 'void', 'call', 'object', 'throw', 'final', 'transact', 'tri', 'statement', 'debug', 'sourc', 'info', 'collect', 'ltproperti', 'main', 'cfghbmbinder'],
            'lucene':['failur', 'case', 'implement', 'throw', 'except', 'instead', 'check', 'junit4', 'gt', 'note', 'ant', 'test', 'gt', 'method', 'caus', 'also', 'follow', 'ad', 'index', 'time', 'document', 'field', 'code', 'new', 'chang', 'fix', 'issu', 'fail', 'use', 'error', 'one', 'creat', 'point', 'see', 'return', 'current', 'problem', 'tri', 'lucen', 'class', 'add', 'reader', 'differ', 'need', 'remov', 'patch',  'public', 'final', 'call', 'sinc', 'support', 'queri', 'first', 'line', 'filter', 'search', 'think', 'would', 'could', 'segment', 'term', 'store', 'valu', 'allow', 'merg', 'make', 'eg', 'string','token', 'result', 'dont', 'like', 'number', 'hit', 'work', 'get', 'doesnt', 'build', 'way', 'bug', 'run', 'default', 'junit', 'file', 'jvm', 'import', 'int', 'null', 'analyz', 'gener', 'score', 'sort', 'version', 'set', 'thread', 'java', 'match', 'api', 'user', 'directori', 'doc', 'delet', 'gt']
        }

        # train,test data
