# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from numpy import array
import FeatureEngineering.continuous as continuous
import FeatureEngineering.discretize as discretize
from sklearn.linear_model import LogisticRegression
from datetime import datetime,timedelta
from FeatureEngineering.Salary_TYP.src import CreditReportFE
import DataMarket.oraclesql as oraclesql
import os
import Config.DataBaseConfig as DBConfig

path='/home/oracle/PycharmProject/'
os.chdir(path)
###读取薪金贷风险额度模型参数表
index_risk = pd.read_excel('./ModelBuilder/模型参数表/薪金贷风险额度模型参数表.xls',sheetname=None)
bankmodel_coef = float(index_risk['风险模型系数'].ix[index_risk['风险模型系数']['指标']=='行内数据模型权重','系数'])
reportmodel_coef = float(index_risk['风险模型系数'].ix[index_risk['风险模型系数']['指标']=='征信报告模型权重','系数'])
riskmodel_threshold = float(index_risk['准入规则'].ix[index_risk['准入规则']['指标']=='风控模型阈值','阈值'])/100
limit_multiple = float(index_risk['建议额度'].ix[index_risk['建议额度']['指标']=='建议额度比例','系数'])
limit_max = int(index_risk['建议额度'].ix[index_risk['建议额度']['指标']=='建议额度最大值','系数'])
region_rate_df = index_risk['地区额度调节比例']
unitkind_rate_df = index_risk['行业额度调节比例']
occupation_rate_df = index_risk['职业额度调节比例']
headship_rate_df = index_risk['职务额度调节比例']
position_rate_df = index_risk['职称额度调节比例']
edu_rate_df = index_risk['学历额度调节比例']

###创建数据集市Oracle连接对象
ip = DBConfig.ip
user = DBConfig.user
passwd = DBConfig.passwd
db = DBConfig.db
ora_market = oraclesql.OracleHelper(ip,user,passwd,db)

def RiskModel(certid,reportfilepath,pcs_entry):
    ###解析征信报告
    (basic,loan_df,loancard_df,preloancard_df,query_df,special_df,adminpunishment_dict,civiljudge_dict,
    taxarrear_dict,forceexecuation_dict,guarantee_dict,loan_dict,special_dict,personalinfo)= CreditReportFE.CreditReportParse(reportfilepath)
    print('征信报告解析完成')
    idtfno = certid
    ###选取输出特征和逾期账户数量用于准入
    certid = '\''+certid+'\''
    query = '''select certid,custid,fullname,certtype,salarycorpname,salaryacctno,salarybranchcode,salarybranchname,branchname,salarybranchregion,overloanacctnum from SALARY_PL_FEATURE where certid = ''' + certid
    df_chunk = ora_market.read_sql(query)
    df_list = []
    for chunk in df_chunk:
        df_list.append(chunk)
    ######################
    # if not df_list:
    #     RiskModelOutput = {}
    #     return RiskModelOutput
    #######################

    RiskModelOutput = df_list[0]
    RiskModelOutput['overloanacctnum'] = RiskModelOutput['overloanacctnum'].fillna(0)
    overloanacctnum = RiskModelOutput.ix[0,'overloanacctnum']
    ####################################################
    ###决策引擎
    if pcs_entry['flag'] == 0:
        IsEntry = CreditReportFE.IsEntry(adminpunishment_dict,civiljudge_dict,taxarrear_dict,forceexecuation_dict,guarantee_dict,loan_dict,special_dict,overloanacctnum)
        if IsEntry == '准入':
            RiskModelOutput['isentry'] = 1
        else:
            RiskModelOutput = RiskModelOutput.drop(['overloanacctnum'],axis=1)
            RiskModelOutput['isentry'] = 0
            RiskModelOutput['businesstype'] = '1110180'
            RiskModelOutput['porttype'] = '02I'
            RiskModelOutput['certid'] = certid
            RiskModelOutput['feature_first'] = IsEntry
            RiskModelOutput['savetime'] = datetime.now().strftime('%Y%m%d %H:%M:%S')
            RiskModelOutput['certtype'] = RiskModelOutput['certtype'].map(lambda x: 'Ind01')
            ora_market.to_sql(RiskModelOutput, 'salary_pl_riskmodeloutput', if_exists='append')
            RiskModelOutput = RiskModelOutput.replace(np.nan, '')
            RiskModelOutput = RiskModelOutput.drop(['savetime'], axis=1)
            RiskModelOutput = RiskModelOutput.to_dict(orient='records')[0]
            return RiskModelOutput
    else:
        RiskModelOutput['overloanacctnum']
        RiskModelOutput['isentry'] = 0
        RiskModelOutput['businesstype'] = '1110180'
        RiskModelOutput['porttype'] = '02I'
        RiskModelOutput['certid'] = certid
        RiskModelOutput['feature_first'] = '被执行人'
        RiskModelOutput['savetime'] = datetime.now().strftime('%Y%m%d %H:%M:%S')
        RiskModelOutput['certtype'] = RiskModelOutput['certtype'].map(lambda x: 'Ind01')
        ora_market.to_sql(RiskModelOutput, 'salary_pl_riskmodeloutput', if_exists='append')
        RiskModelOutput = RiskModelOutput.replace(np.nan, '')
        RiskModelOutput = RiskModelOutput.drop(['savetime'], axis=1)
        RiskModelOutput = RiskModelOutput.to_dict(orient='records')[0]
        return RiskModelOutput

    print('是否准入计算成功！')
    ####################################################

    ###选取实时接口返回变量
    RiskModelOutput = RiskModelOutput.reindex(columns=['businesstype','porttype','corpid','terminalid','corpname','branchname','custid','fullname','certtype','certid',
               'salarycorpname', 'salaryacctno' ,'salarybranchname','salarybranchcode','salarybranchregion','inputdate','isentry','riskscore',
                 'feature_first','feature_second','feature_third','feature_fourth','feature_fifth','isloan','creditlimit', 'isadjust'])
    RiskModelOutput['businesstype'] = '1110180'
    RiskModelOutput['porttype'] = '02I'
    print('实时接口返回变量选取成功！')

    (loan_df,loancard_df,preloancard_df,query_df,special_df,loan_df_process,loancard_df_process,preloancard_df_process,query_df_process,special_df_process,Report) =\
    CreditReportFE.CreditReportProcess(basic,loan_df,loancard_df,preloancard_df,query_df,special_df)
    print('征信报告特征计算成功!')

    PersonalInfo = pd.DataFrame.from_dict([personalinfo],orient='columns')
    Report = pd.concat([PersonalInfo,Report],axis = 1)
    ###保存征信报告基本信息和大宽表
    Report['savedate'] = datetime.now().strftime('%Y%m%d %H%M%S')
    ora_market.to_sql(Report,'salary_pl_report',if_exists='append')
    print('征信报告基本信息+大宽表保存成功！')

    ###选取征信报告风险模型变量
    Report = Report.set_index('certid')
    report_feature = ['loanmonth24laststatnumno','loanmonth24curtermpastdue','loanmonth24ncount','loanmonth24stat123no','loanavglength','recent6loanno','recent6loantotallimit','recent9loanno','recent9loantotallimit','recentloantime',
    'type81longestlength','type81_balance_num','type81maxcreditlength','type81avgcreditlength','type81statallno','type81usebaloverlimit','type81useguarant4avglimit','type81usemaxbaloverlimit','inuseloancreditduration',
    'loaninuseguaranteeway4no','loaninuseavgterm','loaninuseavglimit','loanusemaxpastdueamtoverbal','loanuseavgpastdueamtoverbal','loanusemaxactpayamtoverbal','loanuseavgactpayamtoverbal','queryno']
    RiskModelReportFeature = Report[report_feature]

    ###征信报告变量中文名映射
    report_feature_map = {'loanmonth24laststatnumno_w':u'贷款24月内数字的个数', 'loanmonth24curtermpastdue_w':u'贷款当前逾期期数',
                          'loanmonth24ncount_w':u'贷款24月内n的个数', 'loanmonth24stat123no_w':u'贷款24月内123的个数',
                          'loanavglength_w':u'贷款平均使用年限', 'recent6loanno_w':u'最近6个月贷款数量',
                          'recent6loantotallimit_w':u'最近6个月贷款总额度', 'recent9loanno_w':u'最近9个月贷款数量',
                          'recent9loantotallimit_w':u'最近9个月贷款总额度', 'recentloantime_w':u'最近一次贷款距今时长',
                          'type81longestlength_w':u'贷记卡最长使用年限', 'type81_balance_num_w':u'有欠款的贷记卡个数',
                          'type81maxcreditlength_w':u'贷记卡的最大使用年限', 'type81avgcreditlength_w':u'贷记卡平均信用期限',
                          'type81statallno_w':u'贷记卡24月内数字的个数', 'type81usebaloverlimit_w':u'正在使用贷记卡账户当前本金余额占授信额度的比例',
                          'type81useguarant4avglimit_w':u'正在使用贷记卡账户担保方式为信用的平均授信额度', 'type81usemaxbaloverlimit_w':u'正在使用贷记卡账户当前本金余额占授信额度的最大比例',
                          'inuseloancreditduration_w':u'正在使用的贷记卡平均信用年限', 'loaninuseguaranteeway4no_w':u'贷款正在使用担保方式为4的个数',
                          'loaninuseavgterm_w':u'正在使用的贷款平均年限', 'loaninuseavglimit_w':u'正在使用贷款的平均额度',
                          'loanusemaxpastdueamtoverbal_w':u'正在使用贷款当前逾期金额占本金金额的最大比例', 'loanuseavgpastdueamtoverbal_w':u'正在使用贷款当前逾期金额占本金金额的平均比例',
                          'loanusemaxactpayamtoverbal_w':u'正在使用贷款本月实际还款金额占本金金额的最大比例', 'loanuseavgactpayamtoverbal_w':u'正在使用贷款本月实际还款金额占本金金额的平均比例',
                          'queryno_w':u'查询次数'}

    print('征信报告风险模型变量读取成功！')

    ###征信报告风险模型变量离散化映射
    quantiles = [20*i  for i in range(1,5)]
    discretize_clf = discretize.QuantileDiscretizer(feature_names=report_feature, quantiles=quantiles)
    discretize_clf.cuts = {'loanmonth24laststatnumno': array([0.0,0.0,0.0,0.0]),
    'loanmonth24curtermpastdue': array([0.0,0.0,0.0,0.0]),
    'loanmonth24ncount': array([21.0,24.0,47.0,74.0]),
    'loanmonth24stat123no': array([0.0,0.0,0.0,0.0]),
    'loanavglength': array([0.000000,0.750000,1.575189,3.000000]),
    'recent6loanno': array([0.0,0.0,0.0,0.0]),
    'recent6loantotallimit': array([0.0,0.0,0.0,0.0]),
    'recent9loanno': array([0.0,0.0,0.0,1.0]),
    'recent9loantotallimit': array([0.0,0.0,0.0,40000.0]),
    'recentloantime': array([0.000000,0.333333,1.000000,3.833333]),
    'type81longestlength': array([0.000000,3.166667,5.650000,8.000000]),
    'type81_balance_num': array([1.0,1.0,2.0,4.0]),
    'type81maxcreditlength': array([0.000000,3.166667,5.650000,8.000000]),
    'type81avgcreditlength': array([0.000000,1.792308,3.000000,4.333333]),
    'type81statallno': array([0.0,0.0,0.0,1.0]),
    'type81usebaloverlimit': array([0.299594,0.507450,0.726636,1.006018]),
    'type81useguarant4avglimit': array([10000.000000,17500.000000,26666.666667,43216.446667]),
    'type81usemaxbaloverlimit': array([0.690106,0.998875,1.205557,2.127242]),
    'inuseloancreditduration': array([0.916667,3.000000,9.281818,15.000000]),
    'loaninuseguaranteeway4no': array([0.0,0.0,0.0,1.0]),
    'loaninuseavgterm': array([0.916667,3.000000,9.281818,15.000000]),
    'loaninuseavglimit': array([100000.0,175000.0,261781.4,460000.0]),
    'loanusemaxpastdueamtoverbal': array([0.0,0.0,0.0,0.0]),
    'loanuseavgpastdueamtoverbal': array([0.0,0.0,0.0,0.0]),
    'loanusemaxactpayamtoverbal': array([0.005760,0.007659,0.015000,0.048135]),
    'loanuseavgactpayamtoverbal': array([0.005350,0.007071,0.013098,0.034387]),
    'queryno': array([1.0,2.0,4.0,8.0])
    }

    RiskModelReportFeature_woe = discretize_clf.transform(RiskModelReportFeature)

    print('征信报告风险模型变量离散化映射成功！')

    ###征信报告风险模型变量WOE映射
    woe_clf = continuous.WoeContinuous(feature_names=report_feature)
    woe_clf.maps = {'loanmonth24laststatnumno': {0: -0.203480,4: 1.001788,-1: 0.014797},
    'loanmonth24curtermpastdue': {0: -0.058545,4: 2.440936,-1: -0.149353},
    'loanmonth24ncount': {0: 0.565304,1: -0.557638,2: -0.143735,3: -0.199093,4: -0.056405,-1: 0.014797},
    'loanmonth24stat123no': {0: -0.203740,4: 1.004294,-1: 0.014797},
    'loanavglength': {0: 0.088312,1: 0.378636,2: 0.195640,3: -0.220805,4: -0.651265,-1: -0.623789},
    'recentloantime': {0: 0.168632,1: 0.576309,2: -0.326538,3: -0.171516,4: -0.583711,-1: -0.623789},
    'recent6loanno': {0: -0.189619,4: 0.617603,-1: -0.623789},
    'recent6loantotallimit': {0: -0.189619,4: 0.617603,-1: -0.623789},
    'recent9loanno': {0: -0.256190,3: 0.429292,4: 0.828035,-1: -0.623789},
    'recent9loantotallimit': {0: -0.256190,3: 1.094543,4: 0.416733,-1: -0.623789},
    'loaninuseguaranteeway4no': {0: -0.165919,3: 0.276056,4: 1.003117,-1: -0.149353},
    'loaninuseavgterm': {0: -0.447521,1: 0.772246,2: 0.445802,3: -0.100611,4: -0.570143,-1: -0.149353},
    'type81longestlength': {0: 0.114999,1: 0.488224,2: -0.063742,3: -0.171485,4: -0.618058,-1: -0.623789},
    'type81_balance_num': {0: -0.225321,2: -0.123525,3: 0.067138,4: 0.785944,-1: -0.289190},
    'inuseloancreditduration': {0: -0.447521,1: 0.772246,2: 0.445802,3: -0.100611,4: -0.570143,-1: -0.149353},
    'type81maxcreditlength': {0: 0.114999,1: 0.488224,2: -0.063742,3: -0.171485,4: -0.618058,-1: -0.623789},
    'type81avgcreditlength': {0: 0.061195,1: 0.527763,2: -0.062320,3: -0.287782,4: -0.497809,-1: -0.623789},
    'type81statallno': {0: -0.191856,3: 0.287603,4: 0.655458,-1: -0.249961},
    'queryno': {0: -0.754850,1: -0.846260,2: -0.471720,3: -0.004144,4: 1.042747,-1: -0.623789},
    'type81usebaloverlimit': {0: -0.607660,1: -0.580273,2: 0.290718,3: 0.164483,4: 0.565524,-1: -0.289190},
    'type81useguarant4avglimit': {0: 0.597726,1: 0.241916,2: -0.160357,3: -0.422139,4: -0.327702,-1: -0.289190},
    'type81usemaxbaloverlimit': {0: -0.493460,1: -1.056846,2: 0.492449,3: 0.225851,4: 0.420723,-1: -0.289190},
    'loaninuseavglimit': {0: 0.697967,1: 0.340141,2: -0.417509,3: -0.582057,4: -0.118053,-1: -0.149353},
    'loanusemaxpastdueamtoverbal': {0: -0.058802,4: 2.456361,-1: -0.149353},
    'loanuseavgpastdueamtoverbal': {0: -0.058802,4: 2.456361,-1: -0.149353},
    'loanusemaxactpayamtoverbal': {0: 0.069358,1: -0.691531,2: -0.126897,3: 0.420521,4: 0.481992,-1: -0.149353},
    'loanuseavgactpayamtoverbal': {0: 0.283191,1: -0.775897,2: -0.142933,3: 0.354715,4: 0.420098,-1: -0.149353}
    }

    RiskModelReportFeature_woe = woe_clf.transform(RiskModelReportFeature_woe)
    RiskModelReportFeature_woe.columns = [column+'_w' for column in report_feature]

    print('征信报告风险模型变量WOE映射成功！')

    ###计算征信报告风险模型客户分数
    lr = LogisticRegression(penalty='l2',C=0.1)
    lr.intercept_ = 0.05
    lr.coef_ = array([[ 0.805843  ,  0.52623633,  0.50449108,  0.50449108, -0.2591545,
                        0.15053336,  0.39138835,  0.11274602,  -0.68865785,  0.19428968,
                        0.19428968,  0.38272742,  0.39104119, -0.05127543,  0.65728531,
                        -0.24139594,  -0.35533574,  0.42918568,  0.42918568,  0.07926816,
                        0.07926816,  0.06918716,  0.63320491,  0.3649055 ,  0.00860105,
                        -0.16040549,  0.00615353]])

    RiskModelReportFeature_woe['reportscore'] = lr.predict_proba(RiskModelReportFeature_woe)[:,0]

    ###合并征信报告风险模型变量
    RiskModelReportFeature = pd.concat([RiskModelReportFeature,RiskModelReportFeature_woe],axis=1)

    print('征信报告风险模型客户分数计算成功！')

    ###计算征信报告风险模型变量减分项
    RiskModelReportFeature_imp = RiskModelReportFeature_woe[RiskModelReportFeature_woe.columns[:-1]]*(lr.coef_[0])
    print('征信报告风险模型变量减分项计算成功！')

    ###选取行内风险模型变量
    query_bank = '''select certid,salaryamtsum,custdur,salarynumavg,salarytimegapavg,transoutmonthperamtavg,transoutmonthnumavg,transinmonthnumavg,transmonthnumrateavg,curacctamtsum,intseasonamtavg,consumeloanamtsum,consumeloanbalsum,consumeloanbalrateavg,creditloanamtsum,staffloancontractnum,staffunclearloancontractnum,staffloancontractvaluesum,staffguacontractnum from SALARY_PL_FEATURE  where certid =''' +certid
    df_chunk = ora_market.read_sql(query_bank)
    df_list = []
    for chunk in df_chunk:
        df_list.append(chunk)
    df_salary_feature = df_list[0]
    df_salary_feature = df_salary_feature.set_index('certid')

    bank_feature = ['custdur','salarynumavg','salarytimegapavg','transoutmonthperamtavg','transoutmonthnumavg','transinmonthnumavg','transmonthnumrateavg','curacctamtsum','intseasonamtavg','consumeloanamtsum','consumeloanbalsum','consumeloanbalrateavg','creditloanamtsum','staffloancontractnum','staffunclearloancontractnum','staffloancontractvaluesum','staffguacontractnum']
    RiskModelBankFeature = df_salary_feature[bank_feature]
    RiskModelBankFeature = RiskModelBankFeature.fillna(0)
    print('行内风险模型变量选取成功！')

    ###行内数据变量中文名映射
    bank_feature_map = {'custdur_w':u'开户时长', 'salarynumavg_w':u'工资笔数平均值',
                        'salarytimegapavg_w':u'工资时间间隔平均值', 'transoutmonthperamtavg_w':u'月单笔交易往账金额平均值',
                        'transoutmonthnumavg_w':u'月交易往账笔数平均值', 'transinmonthnumavg_w':u'月交易来账笔数平均值',
                        'transmonthnumrateavg_w':u'交易来往账月交易笔数比例平均值', 'curacctamtsum_w':u'活期储蓄账户总积数',
                        'intseasonamtavg_w':u'存息收入季度金额平均值', 'consumeloanamtsum_w':u'消费贷款总金额',
                        'consumeloanbalsum_w':u'消费正常本金总金额', 'consumeloanbalrateavg_w':u'消费正常本金金额占比平均值',
                        'creditloanamtsum_w':u'信用担保贷款总金额', 'staffloancontractnum_w':u'担保合同个数',
                        'staffunclearloancontractnum_w':u'未结清贷款担保合同个数', 'staffloancontractvaluesum_w':u'担保合同总价值',
                        'staffguacontractnum_w':u'担保合同个数'}
    bank_feature_map.update(report_feature_map)
    feature_map = bank_feature_map

    ###行内风险模型变量离散化映射
    #WOE计算
    quantiles = [20*i  for i in range(1,5)]
    discretize_clf = discretize.QuantileDiscretizer(feature_names=bank_feature, quantiles=quantiles)
    discretize_clf.cuts = {'consumeloanamtsum': array([0.,100000.,200000.,400000.]),
    'consumeloanbalrateavg': array([0.,0.,0.42857143,0.85714286]),
    'consumeloanbalsum': array([0.,0.,100000.,200000.]),
    'creditloanamtsum': array([0.,0.,0.,0.]),
    'curacctamtsum': array([0.,835.65,10826.59,64875.44]),
    'custdur': array([968.,1548.,2271.,3506.]),
    'intseasonamtavg': array([3.171,6.84357143,12.614,21.77111111]),
    'salarynumavg': array([1.31034483,1.71428571,2.2,2.51724138]),
    'salarytimegapavg': array([11.38461538,13.13846154,17.06451613,24.62857143]),
    'staffguacontractnum': array([0.,0.,2.,5.]),
    'staffloancontractnum': array([0.,0.,4.,9.]),
    'staffloancontractvaluesum': array([0.,0.,700000.,1760000.]),
    'staffunclearloancontractnum': array([0.,0.,2.,4.]),
    'transinmonthnumavg': array([1.,1.5,1.88235294,2.7]),
    'transmonthnumrateavg': array([0.03880071,0.12690126,0.2260101,0.39937831]),
    'transoutmonthnumavg': array([2.6,3.81818182,5.46428571,7.4]),
    'transoutmonthperamtavg': array([1316.52503423,2351.66490533,3669.28015261,6883.14931602])}
    RiskModelBankFeature_woe = discretize_clf.transform(RiskModelBankFeature)

    print('行内风险模型变量离散化映射成功！')

    ###行内风险模型变量WOE映射
    woe_clf = continuous.WoeContinuous(feature_names=bank_feature)
    woe_clf.maps = {'consumeloanamtsum': {0: 0.099717665435039987,1: -1.4755213859708101,2: -0.25609483774303393,3: 0.53738960204021413,4: -0.0080537348070968268},
    'consumeloanbalrateavg': {0: 0.089710433958794786,2: -0.22463988557514747,3: 0.53738960204021413,4: -0.86371984486481701},
    'consumeloanbalsum': {0: 0.089710433958794786,2: -1.4233356328002398,3: 0.21325326869981145,4: 0.58711303739264475},
    'creditloanamtsum': {0: 0.16451164704149313, 4: -2.2182655076701274},
    'curacctamtsum': {0: 0.36518234250364429,1: 0.19121738030063096,2: -0.3554950027988838,3: -0.86371984486481701,4: 0.19715211582044559},
    'custdur': {0: -0.2876820724517809,1: -2.0794415416798357,2: 0.5045560107523952,3: -0.12405264866997888,4: 0.42933258951480774},
    'intseasonamtavg': {0: 0.3542392014358588,1: 0.10331608601457336,2: -0.66881950586400229,3: -0.3554950027988838,4: 0.28524632994890731},
    'salarynumavg': {0: 0.24386111378605296,1: 0.55242747940475467,2: -0.66881950586400229,3: 0.36842383642781523,4: -1.3992380812211793},
    'salarytimegapavg': {0: -1.1115560087693983,1: 0.36842383642781523,2: -0.22463988557514747,3: 0.36842383642781523,4: 0.10331608601457336},
    'staffguacontractnum': {0: -0.066368886774869212,2: -0.30410405692939446,3: 0.39741137330106746,4: -0.083561287315241844},
    'staffloancontractnum': {0: -0.056334474615027968,2: -0.3127471530854144,3: -0.43568041471811364,4: 0.68163921088476087},
    'staffloancontractvaluesum': {0: -0.056334474615027968,2: -2.4218644629113668,3: -0.51466882603674402,4: 1.0502072583374455},
    'staffunclearloancontractnum': {0: -0.085535621618959598,2: -1.3492276606465179,3: -0.23119728612130655,4: 1.1897462104117582},
    'transinmonthnumavg': {0: -0.17057266430487161,1: 0.41877772257343004,2: -0.45825473675665274,3: 0.10331608601457336,4: -0.10603414316730052},
    'transmonthnumrateavg': {0: -0.36839840763479165,1: 0.44735109501748604,2: -0.10603414316730052,3: 0.19715211582044559,4: -0.3554950027988838},
    'transoutmonthnumavg': {0: 0.32645963732878314,1: -0.079005470779381193,2: -0.22463988557514747,3: 0.28524632994890731,4: -0.50192980025931433},
    'transoutmonthperamtavg': {0: 0.3542392014358588,1: -0.86371984486481701,2: 0.0026990569691649835,3: -0.3554950027988838,4: 0.44735109501748604}}

    RiskModelBankFeature_woe = woe_clf.transform(RiskModelBankFeature_woe)
    RiskModelBankFeature_woe.columns = [column+'_w' for column in bank_feature]

    print('行内风险模型变量WOE映射成功！')

    ###计算行内风险模型客户分数
    lr = LogisticRegression(penalty='l2',C=0.1)
    lr.intercept_ = array([-1.36577499])
    lr.coef_ = array([[ 0.32917076,  0.45498924,  0.11491545,  0.41759412,  0.12258524,
         0.22386615,  0.21834257,  0.29242046,  0.26968874,  0.22281269,
         0.14474189,  0.35161893,  0.53438019,  0.14997334,  0.21574959,
         0.38207319,  0.0832601 ]])
    RiskModelBankFeature_woe['bankscore'] = lr.predict_proba(RiskModelBankFeature_woe)[:,0]

    ###合并行内风险模型变量
    RiskModelBankFeature = pd.concat([RiskModelBankFeature,RiskModelBankFeature_woe],axis=1)
    print('行内风险模型客户分数计算成功！')

    ###计算行内风险模型变量减分项
    RiskModelBankFeature_imp = RiskModelBankFeature_woe[RiskModelBankFeature_woe.columns[:-1]]*(lr.coef_[0])
    print('行内风险模型变量减分项计算成功！')

    ###合并行内和征信报告风险模型变量
    #RiskModelBankFeature = RiskModelBankFeature.reset_index()
    RiskModelReportFeature = RiskModelReportFeature.reset_index()
    RiskModelReportFeature = RiskModelReportFeature.drop(['certid'], axis=1)
    RiskModelBankFeature = RiskModelBankFeature.reset_index()
    RiskModelFeature = pd.concat([RiskModelBankFeature,RiskModelReportFeature],axis=1)
    RiskModelOutput['riskscore'] = round(bankmodel_coef*RiskModelBankFeature.ix[0,'bankscore']+reportmodel_coef*RiskModelReportFeature.ix[0,'reportscore'],2)
    RiskModelOutput.ix[RiskModelOutput['riskscore']>riskmodel_threshold,'isloan'] = 1
    RiskModelOutput['isloan'] = RiskModelOutput['isloan'].fillna(0)
    RiskModelOutput['isloan'] = RiskModelOutput['isloan'].astype(int)
    print('风险评分+是否放贷计算成功！')

    ###合并行内和征信报告风险模型变量减分项
    RiskModelReportFeature_imp = RiskModelReportFeature_imp.reset_index()
    RiskModelReportFeature_imp = RiskModelReportFeature_imp.drop(['certid'], axis=1)
    RiskModelBankFeature_imp = RiskModelBankFeature_imp.reset_index()
    RiskModelBankFeature_imp = RiskModelBankFeature_imp.drop(['certid'], axis=1)
    #RiskModelBankFeature_imp = RiskModelBankFeature_imp.reset_index()
    #RiskModelBankFeature_imp = RiskModelBankFeature_imp.set_index('certid')
    RiskModelFeature_imp = pd.concat([RiskModelBankFeature_imp,RiskModelReportFeature_imp],axis=1)

    RiskModelOutput['feature_first'] = feature_map.get(list(RiskModelFeature_imp.ix[RiskModelFeature_imp.first_valid_index()].sort_values().index)[0])
    RiskModelOutput['feature_second'] = feature_map.get(list(RiskModelFeature_imp.ix[RiskModelFeature_imp.first_valid_index()].sort_values().index)[1])
    RiskModelOutput['feature_third'] = feature_map.get(list(RiskModelFeature_imp.ix[RiskModelFeature_imp.first_valid_index()].sort_values().index)[2])
    RiskModelOutput['feature_fourth'] = feature_map.get(list(RiskModelFeature_imp.ix[RiskModelFeature_imp.first_valid_index()].sort_values().index)[3])
    RiskModelOutput['feature_fifth'] = feature_map.get(list(RiskModelFeature_imp.ix[RiskModelFeature_imp.first_valid_index()].sort_values().index)[4])
    print('重要减分项计算成功！')

    ###计算建议额度
    region_rate = float(region_rate_df.ix[region_rate_df['地区']==RiskModelOutput.ix[0,'salarybranchregion'],'比例'])
    unitkind_rate = float(unitkind_rate_df.ix[unitkind_rate_df['行业代码']==Report.ix[0,'unitkind'],'比例'])
    occupation_rate = float(occupation_rate_df.ix[occupation_rate_df['职业代码']==Report.ix[0,'occupation'],'比例'])
    headship_rate = float(headship_rate_df.ix[headship_rate_df['职务代码']==Report.ix[0,'headship'],'比例'])
    position_rate = float(position_rate_df.ix[position_rate_df['职称代码']==Report.ix[0,'position'],'比例'])
    edu_rate = float(edu_rate_df.ix[edu_rate_df['学历代码']==Report.ix[0,'eduexperience'],'比例'])
    if RiskModelOutput.ix[0,'isloan'] == 1:
        RiskModelOutput['creditlimit'] = min(df_salary_feature.ix[0,'salaryamtsum']*limit_multiple*region_rate*unitkind_rate*occupation_rate*headship_rate*position_rate*edu_rate*RiskModelOutput.ix[0,'riskscore']//1000*1000,limit_max)
    print('建议额度计算成功！')

    ###保存实时接口结果
    RiskModelOutput['savetime'] = datetime.now().strftime('%Y%m%d %H:%M:%S')
    RiskModelOutput['certtype'] = RiskModelOutput['certtype'].map(lambda x:'Ind01')
    ora_market.to_sql(RiskModelOutput,'salary_pl_riskmodeloutput',if_exists='append')
    print('实时接口结果保存成功！')

    ###保存风险模型变量
    RiskModelFeature['savetime'] = datetime.now().strftime('%Y%m%d %H:%M:%S')
    #RiskModelFeature = RiskModelFeature.reset_index()
    ora_market.to_sql(RiskModelFeature,'salary_pl_riskmodelfeature',if_exists='append')
    print('风险模型变量保存成功！')

    RiskModelOutput = RiskModelOutput.replace(np.nan,'')
    RiskModelOutput = RiskModelOutput.drop(['savetime'],axis=1)
    RiskModelOutput = RiskModelOutput.to_dict(orient = 'records')[0]
    return RiskModelOutput

if __name__ == '__main__':
    #certid = '410303196910240513'
    certid2 = '410324199005013113'
    certid3 = '41032419900628031X'
    certid4 = '41032419890309001X'
    certid5 = '410324199001253726'
    certid =  '41032419900628031X'
    reportfilepath = '/home/oracle/PycharmProject/ServerConnector_stc/Tcp/Reportfile/41032419890309001X_2017081400000007.json'
    #reportfilepath = '/home/oracle/PycharmProject/FeatureEngineering/Salary_TYP/reportfiles/任勇民.txt'
    reportfilepath2= '/home/oracle/PycharmProject/FeatureEngineering/Salary_TYP/reportfiles/代振南.txt'
    reportfilepath3 = '/home/oracle/PycharmProject/FeatureEngineering/Salary_TYP/reportfiles/吴兴源.txt'
    reportfilepath4 = '/home/oracle/PycharmProject/FeatureEngineering/Salary_TYP/reportfiles/王民益.txt'
    reportfilepath5 = '/home/oracle/PycharmProject/FeatureEngineering/Salary_TYP/reportfiles/李闪闪.txt'
    pcs_entry = {'flag':0}
    res = RiskModel(certid,reportfilepath,pcs_entry)
