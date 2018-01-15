import numpy
import pandas

data = pandas.read_csv(
    'D:\\PDA\\5.7\\data.csv'
)

data['DealDateTime'] = pandas.to_datetime(
    data.DealDateTime, 
    format='%Y/%m/%d'
)

data['DateDiff'] = pandas.to_datetime(
    'today'
) - data['DealDateTime']

data['DateDiff'] = data['DateDiff'].dt.days

R_Agg = data.groupby(
    by=['CustomerID']
)['DateDiff'].agg({
    'RecencyAgg': numpy.min
})

F_Agg = data.groupby(
    by=['CustomerID']
)['OrderID'].agg({
    'FrequencyAgg': numpy.size
})

M_Agg = data.groupby(
    by=['CustomerID']
)['Sales'].agg({
    'MonetaryAgg': numpy.sum
})

aggData = R_Agg.join(F_Agg).join(M_Agg)

bins = aggData.RecencyAgg.quantile(
    q=[0, 0.2, 0.4, 0.6, 0.8, 1],
    interpolation='nearest'
)
bins[0] = 0
labels = [5, 4, 3, 2, 1]
R_S = pandas.cut(
    aggData.RecencyAgg, 
    bins, labels=labels
)

bins = aggData.FrequencyAgg.quantile(
    q=[0, 0.2, 0.4, 0.6, 0.8, 1],
    interpolation='nearest'
)
bins[0] = 0;
labels = [1, 2, 3, 4, 5];
F_S = pandas.cut(
    aggData.FrequencyAgg, 
    bins, labels=labels
)

bins = aggData.MonetaryAgg.quantile(
    q=[0, 0.2, 0.4, 0.6, 0.8, 1],
    interpolation='nearest'
)
bins[0] = 0
labels = [1, 2, 3, 4, 5]
M_S = pandas.cut(
    aggData.MonetaryAgg, 
    bins, labels=labels
)

aggData['R_S']=R_S
aggData['F_S']=F_S
aggData['M_S']=M_S

aggData['RFM'] = 100*R_S.astype(int) + 10*F_S.astype(int) + 1*M_S.astype(int)

bins = aggData.RFM.quantile(
    q=[
        0, 0.125, 0.25, 0.375, 0.5, 
        0.625, 0.75, 0.875, 1
    ],
    interpolation='nearest'
)
bins[0] = 0
labels = [1, 2, 3, 4, 5, 6, 7, 8]
aggData['level'] = pandas.cut(
    aggData.RFM, 
    bins, labels=labels
)

aggData = aggData.reset_index()

aggData.sort(
    ['level', 'RFM'], 
    ascending=[1, 1]
)

aggData.groupby(
    by=['level']
)['CustomerID'].agg({
    'size':numpy.size
})
