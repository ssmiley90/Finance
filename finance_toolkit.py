import logging
import math
import numpy as np
import pandas as pd
import re
import plotly
import scipy.stats as stats
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import copy
sizeOfPortfolio = 5000*10 #5/000/000/000/000

# %%
def init_config(loggingFileAddress):
    loggerFormat = "%(asctime)s %(message)s "
    logging.basicConfig(loggingFileAddress, format=loggerFormat, filemode="w")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Anything changes makes the code to reload if you import your own code, then modify it after loading it to your
    # main code, then the loaded code does not modify in- -your code unless you delete and re-import it again. by
    # setting the autoreload extension to number 2 every time you - modify the loaded code the main code
    # automatically reloads and updates and modifies it to.

    #%load_ext autoreload #these two line should be added in jupyter mode
    #%autoreload 2


# %%
def input_rawdata(fileAddress, dateFormat="%Y%m", toPer='M'):
    import os
    split_tup = os.path.splitext(fileAddress)
    print(split_tup)
    # extract the file name and extension

    file_name = split_tup[0]
    file_extension = split_tup[1]

    if file_extension == ".csv":
        print("file extension: CSV")
        #inputData = pd.read_csv(fileAddress, header=0, index_col=0, parse_dates=True ,format=dateFormat )

        inputData = pd.read_csv(fileAddress,index_col=[0],header=[0],parse_dates=True)
        inputData = inputData.backfill()
        #inputData.index.to_period(toPer)
        inputData.index=pd.to_datetime(inputData.index, format=dateFormat).to_period(toPer)

    elif file_extension == ".xlsx" :
        print("file extension: XLSX")
        # inputData = pd.read_csv(fileAddress, header=0, index_col=0, parse_dates=True ,format=dateFormat )

        inputData = pd.read_excel(fileAddress, index_col=[0], header=[0], parse_dates=True)
        inputData = inputData.backfill()
        # inputData.index.to_period(toPer)
        inputData.index = pd.to_datetime(inputData.index, format=dateFormat).to_period(toPer)

    return inputData


# %%
def convertFreq(inputData,func,period='M'):
    '''
    This function receives an inputData and returns the casted data to given period
    :param inputData: a pandas DataFrame
    :param period:
    :return:
    '''
    return inputData.groupby(inputData.index.to_timestamp(freq=period)).apply(func)

# %%
def ann_ret(periodReturn, period=None, periodType=None):
    compoundingFactor = 0
    periodType = periodReturn.index.freq.name
    if isinstance(periodReturn, pd.DataFrame) or isinstance(periodReturn, pd.Series):
        period = periodReturn.shape[0]
        periodReturn = (1 + periodReturn).prod() - 1


    if periodType == 'M':
        compoundingFactor = 12 / period
    elif periodType == 'D':
        compoundingFactor = 365 / period
    elif periodType == 'Y':
        compoundingFactor = 1 / period
    else:
        pass
    return (1 + periodReturn) ** compoundingFactor - 1


# %%
def ann_vol(periodVol, period=None, periodType=None):
    '''

    :param periodVol: a data series
    :param period: period of data
    :param periodType: period type one of M D Y
    :return: (a,b)  which a represents standard deviation of each series annualized and b is cov matrix of whole series
    which were annualized. Hence, a^2= var(each series)  and b^2 = COV(data series )
    '''
    compoundingFactor = 0

    if isinstance(periodVol, pd.DataFrame) or isinstance(periodVol, pd.Series):
        if periodType is None :
            periodType = periodVol.index.freq.name
        period = periodVol.shape[0]
        covMatRoot=np.sqrt ( periodVol.cov() )
        periodVol = (periodVol.std())

    # print(periodVol , periodType , period)
    if periodType == 'M':
        compoundingFactor = 12
    elif periodType == 'D':
        compoundingFactor = 365
    elif periodType == 'Y':
        compoundingFactor = 1

    return ( periodVol * np.sqrt(compoundingFactor) , covMatRoot * np.sqrt(compoundingFactor) )


# %%
def prc2ret(prcSeries):
    # this function get a pd.df.series and return the RETURN (change percentages) by 3 different ways.
    # first method :  using shift function of pd.df
    # sec method : using pct_change function of pd.df
    # third method : using .values attr of pd.df
    # recall that x[1:]/ x[:-1]  does not work because when you divide two different pd.series, python match these two series based index column.
    fOutput = (
        prcSeries / prcSeries.shift(1) - 1, prcSeries.pct_change(), prcSeries[1:].values / prcSeries[:-1].values - 1)
    return fOutput[0]


# %%
def Corr2Cov(corrMat, StandardDeviations):
    '''

    :param corrMat:
    :param StandardDeviations:
    :return:
    '''

    #print("corr",corrMat.shape)
    #print("std1",StandardDeviations.shape)
    if isinstance(StandardDeviations,pd.Series):
        StandardDeviations=StandardDeviations.to_numpy()
        StandardDeviations = StandardDeviations[np.newaxis]
    #print("std2", StandardDeviations.shape)
    return np.multiply(StandardDeviations @ StandardDeviations.T, corrMat)


# %%
def cmpt_return(retSeries):
    '''
    :param retSeries:
    :return:
    '''
    # compute expected return based on historic returns
    #input:
    #output:
    return np.expm1(np.log1p(retSeries).sum())
    # return (retSeries + 1).prod() - 1

# %%
def cmpt_time2liq (w,tradeVol):
    '''
    :param w: Weight of Stocks in portfolio
    :param tradeVol: Value of transactions on each stock based on Billion Rials
    :return: Compute Days remaining to liquify based on Average Vol and Value of each stock in our portfolio
    '''
    return np.ceil( (w * sizeOfPortfolio) / tradeVol )

# %%
def cmpt_vol(retSeries,time2liq=None):
    # We define volatility as standard deviation of return series
    # Mode : {0:"All the methods" , 1 : "Std_Sample" , 2:"Std_Population" , 3:"Semi Std" , 4: "Liquidity adjusted" }
    # retSeries=cmpt_return(prcSeries)
    if time2liq is None:
        time2liq= np.ones(retSeries.columns.size)

    time2liq=np.sqrt( ((2*time2liq+1)*(time2liq+1))/(6*time2liq)  )

    #should be defined formula of time2liq to adjesting factor
    stdSample = retSeries.std()* time2liq
    stdPopulation = retSeries.std(ddof=0)* time2liq  # ddof is Degree of freedom;the divisor is N-ddf ; default value of ddf is 1
    semiStd = retSeries[retSeries < retSeries.mean()].std()* time2liq
    stdAdj= stdSample * time2liq # SHOULD BE COMPELETED
    return stdSample, stdPopulation, semiStd,stdAdj


# %%
def cmpt_sharpRatio(retSeries, retVar=None, rfRate=.0):
    # retSeries=cmpt_return(prcSeries)
    if isinstance(retSeries, pd.DataFrame) or isinstance(retSeries, np.ndarray):
        return ((1 + retSeries).prod() - rfRate) / cmpt_vol(retSeries)[0]
    else:
        return (retSeries - rfRate) / retVar


# %%
def portfolio_sharpRatio(W, data, rf , time2liq=None):
    return (-1 * cmpt_sharpRatio(portfolio_expected_return(W=W, Rets=None, data=data),
                                 portfolio_expected_risk(W=W, standardDevs=None, corrMat=None, data=data,time2liq=time2liq), rf))


# %%
def cmpt_cov_martix(retSeries, windowlen):
    pass


# %%
def max_drwdwn(retSeries):
    # compute maximum draw down
    # retSeries=compt_return(prcSeries)
    wealth = (1 + retSeries).cumprod()
    latestGlobalPick = (1 + retSeries).cumprod().cummax()
    maxDrwdwn = wealth / latestGlobalPick - 1
    return pd.DataFrame({"wealth": wealth, "latestPick": latestGlobalPick, "maxDrwdwn": maxDrwdwn})


# %%
def VaR(retSeries, confidenceLevel=0.95, mode="parametric",time2liq=None):
    # mode = historic / parametric / modified /
    # isinstance ( obj , obj ) to check weather input one is an instance of object 2 or not

    if mode == "parametric":
        return retSeries.mean() + stats.norm.ppf(1 - confidenceLevel) * cmpt_vol(retSeries)[0]
    elif mode == "historic":
        return retSeries.quantile(1 - confidenceLevel)
    elif mode == "modified":
        z = stats.norm.ppf(confidenceLevel)  # Based on a formule by CORNISH-FISHER
        zModif = z + (1 / 6) * (z ** 2 - 1) + (1 / 24) * (z ** 3 - 3 * z) * (stats.kurtosis(retSeries) - 3) - (
                1 / 36) * (2 * z ** 3 - 5 * z) * (stats.skew(retSeries) ** 2)
        return retSeries.mean() + zModif * cmpt_vol(retSeries)[0]
    if mode == "liqAdj":
        return retSeries.mean() + stats.norm.ppf(1 - confidenceLevel) * cmpt_vol(retSeries,time2liq)[3] #None should be completed
    elif mode == "garch":
        pass


# %%
def portfolio_expected_return(W, Rets, data=None ):
    # if isinstance(data,pd.DataFrame) or isinstance(data,np.ndarray):
    if data is not None:
        Rets = (1 + data).prod() - 1
        return np.dot(Rets, W)
    else:
        return np.dot(Rets, W)


# %%
def portfolio_expected_risk(W, standardDevs, corrMat, data=None, tradeVol=None):
    time2liq=None
    if tradeVol is not None :
        time2liq=cmpt_time2liq(W,tradeVol)

    if data is not None:
        standardDevs = cmpt_vol(data,time2liq)[0] # standardDevs= comp_vol( data , time2liq)    | last version : data.std()
        corrMat = data.corr()
    if isinstance(standardDevs,pd.Series):
        standardDevs=standardDevs.to_numpy()
        standardDevs=standardDevs[np.newaxis]
        standardDevs=standardDevs.T
    covMat =  Corr2Cov(corrMat,standardDevs)
    # for i in range(corrMat.shape[0]):
    #     covMat.iloc[i, :] *= standardDevs[i]
    # for i in range(corrMat.shape[1]):
    #     covMat.iloc[:, i] *= standardDevs[i]
    # print("in portfolio function:", corrMat)
    # return np.sqrt(np.dot(np.dot(W, covMat), W))
    return (W.T @ covMat @ W) ** 0.5

# %%
#
#   portfolio_expected_VaR
def portfolio_expected_VaR (VaR , corrMatt , data=None ) :
    '''

    :param VaR:
    :param corrMatt:
    :param data:
    :return:
    '''
    if data is not None:
        #standardDevs = data.std()
        corrMat = data.corr()
    '''
    if isinstance(standardDevs,pd.Series):
        standardDevs=standardDevs.to_numpy()
        standardDevs=standardDevs[np.newaxis]
    '''

    return (VaR.T @ corrMat @ VaR) ** 0.5

# %%
def eff_point(Ret, corrMat, standardDeviation, targetRet, data=None,tradeVol=None):

    from scipy.optimize import minimize
    #print("eff point func :",standardDeviation.shape)
    # if isinstance(data,pd.DataFrame):
    if data is not None:
        NoR = data.shape[1]  # Number of Records
        limit1 = {"type": "eq", 'fun': lambda x: x.sum() - 1}
        limit2 = {'type': 'eq',
                  'fun': lambda x, Ret, targetRet, data: portfolio_expected_return(x, Ret, data) - targetRet,
                  'args': (None, targetRet, data)}
        res = minimize(portfolio_expected_risk, x0=np.repeat(1 / NoR, NoR), args=(None, None, data,tradeVol),
                       bounds=[(0, 1)] * NoR, constraints=(limit1, limit2), method="SLSQP")
    else:
        NoR= Ret.size
        limit1 = {"type": "eq", 'fun': lambda x: x.sum() - 1}
        limit2 = {'type': 'eq',
                  'fun': lambda x, Ret, targetRet, data: portfolio_expected_return(x, Ret, data) - targetRet,
                  'args': (Ret, targetRet, None)}
        res = minimize(portfolio_expected_risk, x0=np.repeat(1 / NoR, NoR), args=(standardDeviation, corrMat, None,None),
                       bounds=[(0, 1)] * NoR, constraints=(limit1, limit2), method="SLSQP")
    return res


# %%
def eff_frontier(Ret, corrMat, standardDeviation, data=None , tradeVol=None):
    # if isinstance(data, pd.DataFrame) or isinstance(data,pd.Series): # actually it seems that the only correct part
    # is pd.Series so later should be scrutinized

    #print("eff frontier func :", standardDeviation.shape)

    if data is None:
        targetRet = np.linspace(Ret.min(), Ret.max(), 100)
        optimizationOutputs = [eff_point(Ret, corrMat, standardDeviation, i ) for i in targetRet]
    else:
        targetRet = np.linspace(cmpt_return(data).min(), cmpt_return(data).max(), 100)
        optimizationOutputs = [eff_point(None, None, None, i, data , tradeVol) for i in targetRet]

    return optimizationOutputs


# %%
def GMV(corrMat, standardDeviation, data=None, Ret=None,tradeVol=None):
    #   return global minimum volatility portfolio
    covMat = Corr2Cov(corrMat, standardDeviation)
    NoR = covMat.shape[1]
    from scipy.optimize import minimize

    if data is None:
        limits1 = {"type": "eq", 'fun': lambda x: x.sum() - 1}
        res = minimize(portfolio_expected_risk, x0=np.repeat(1 / NoR, NoR), args=(standardDeviation, corrMat),
                       bounds=((0, 1),)*NoR, constraints=(limits1,))
    else:
        limits1 = {"type": "eq", 'fun': lambda x: x.sum() - 1}
        res = minimize(portfolio_expected_risk, x0=np.repeat(1 / NoR, NoR), args=(None,None,data, tradeVol),
                       bounds=((0, 1),) * NoR, constraints=(limits1,))

        #limits1 = np.linspace(cmpt_return(data).min(), cmpt_return(data).max(), 100)
        #res = minimize(portfolio_expected_risk, x0=np.repeat(1 / NoR, NoR), args=(None, None, data, tradeVol),
        #               bounds=[(0, 1)] * NoR, constraints=(limit1, limit2), method="SLSQP")

    return res.x


# %%
def cm_line(rf, data=None):
    NoR = data.shape[1]  # Number of Records
    from scipy.optimize import minimize
    limit1 = {"type": "eq", 'fun': lambda x: x.sum() - 1}
    res = minimize(portfolio_sharpRatio, x0=np.repeat(1 / NoR, NoR), args=(data, rf), bounds=[(0, 1)] * NoR,
                   constraints=limit1)
    return res


# %%

def cppi(periodRet, periodMaxRet, floor=0, cap=0, lossThreshold=1, profitThreshold=1, mode='simple'):
    #
    mLossFactor=0
    if floor != 0:
        mLossFactor = 1 / (lossThreshold - floor)
    if cap != 0:
        mProfitFactor = 1 / (cap - profitThreshold)

    if mode == 'simple':
        return ((1 + periodRet) < lossThreshold) * mLossFactor * ((1 + periodRet) - floor) + (
                (1 + periodRet) > profitThreshold) * mProfitFactor * (cap - (1 + periodRet))
    else:
        return (((1 + periodRet) / (1 + periodMaxRet)) <= lossThreshold) * (
                ((1 + periodRet) / (1 + periodMaxRet)) - floor) * mLossFactor + (
                (1 - (periodMaxRet - periodRet)) > lossThreshold) * 1

# %%


# FUNDING RATIO IS ASSET/LIABILITIES
# HEDGING = CASHFLOW MATCHING
# LIABILITY-HEDGING PORTFOLIOS(LHP) OR GOAL-HEDGING PORTFOLIOS(GHP)
# COX INGERSOLL ROSS MODEL (CIR) FOR INTEREST RATE SIMULATION : dr= a(b-rt)dt + sigma.sqrt(rt).dWt

# Liabilitiy-driven-investing(LDI)   dividing the portfolio between PERFOREMANCE-SEEKING-PORTFOLIO (PSP) AND
# LIABILITIY-HEDGING-PORTFOLIO (LHP)
# MAXIMIZING UTILITY( AT/ LT ) OR MAX U(FUNDING RATIO)
def funding_ratio():
    pass


# %%

def trackingError(r1,r2) :
    return np.sqrt (  ((r1-r2)**2).sum()  )


# %%
def ew_weighting (IndRetData, McWeights, corrMat,stds,**kwargs):
    if 'limits' not in kwargs:
        limits=[0,1]
        # print("limits are not defined")
    else:
        limits=kwargs['limits']

    pos=0
    numberOfIndices = IndRetData.shape[1]
    numberOfDates = IndRetData.shape[0]
    weights = pd.Series( np.repeat(1 / numberOfIndices, numberOfIndices ) )
    weights.index=IndRetData.columns

    if limits[0] != 0:
        weights[McWeights.iloc[pos] < limits[0]] = 0
        weights = weights.divide( weights.sum())

    if limits[1] != 1:
        weights[  weights > limits[1]*McWeights.iloc[pos]  ] = limits[1]*McWeights.iloc[pos]
        weights = weights.divide(weights.sum())

    return weights


# %%
def cw_weighting(IndRetData, McWeights, corrMat,stds,**kwargs):
    # print(McWeights)
    if 'limits' not in kwargs:
        limits=[0,1]
        # print("limits are not defined")
    else:
        limits=kwargs['limits']

    pos=1
    numberOfIndices = IndRetData.shape[1]
    numberOfDates = IndRetData.shape[0]
    weights = pd.Series( np.repeat(0, numberOfIndices ) )
    weights.index=IndRetData.columns
    weights=McWeights.iloc[pos]

    if limits[0] != 0:
        weights[McWeights.iloc[pos] < limits[0]] = 0
        weights = weights.divide(weights.sum())

    if limits[1] != 1:
        weights[weights > limits[1] * McWeights.iloc[pos]] = limits[1] * McWeights.iloc[pos]
        weights = weights.divide(weights.sum())

    return weights

# %%
def gmv_weighting(IndRetData, McWeights, corrMat,stds,**kwargs):
    weights = GMV(corrMat, stds)
    return weights


# %%
def portfolio_stock_weights(IndRetData, IndMcData=None, weighting=ew_weighting,
                            estimation_window=36, **kwargs):
    # methods : EW = 1/n equivalent to Naive  VW: Market Capital Weighted  GW: Global portfolio
    # Limits : [L1,L2] :  L1 -> [Threshold ] means Industries or Stocks with MarketCap less than Threshold
    # will be omitted
    #                     L2 as a multiple -> minimum(L2*MarketCapIndex(i) ,  weight(i ))   if a multiple limit pass to
    #                     function then an upper-bound of MarketCap of each indices times L2 will be applied

    numberOfIndices = IndRetData.shape[1]
    numberOfDates = IndRetData.shape[0]
    weights = pd.DataFrame(
            np.repeat(np.nan, numberOfIndices * numberOfDates).reshape(numberOfDates, numberOfIndices))
    weights.columns = IndRetData.columns
    weights.index = IndRetData.index
    weights = pd.DataFrame().reindex_like(IndRetData)
    weights.columns = IndRetData.columns


    McWeights = pd.DataFrame().reindex_like(IndRetData)
    McWeights.columns = IndRetData.columns
    if IndMcData is not None:
        McWeights = IndMcData.divide(IndMcData.sum(axis='columns'), axis='rows')


    wndws   = [(i, i + estimation_window) for i in range(IndRetData.shape[0] - estimation_window)]
    corrMat = [ cov_estimation( IndRetData.iloc[wins[0]:wins[1]] , **kwargs)[0] for wins in wndws]
    stds    = [IndRetData.iloc[wins[0]:wins[1]].std() for wins in wndws]
    result  = [weighting(IndRetData.iloc[wndws[i][0]:wndws[i][1]],McWeights.iloc[wndws[i][0]:wndws[i][1]],
                        corrMat[i], stds[i] , **kwargs) for i in range(len(wndws))]
    for i in range(len(result)):
        # print(result[i])
        weights.iloc[i + estimation_window] = result[i]

    return weights


# %%
def backtest(retData, weights=None, **kwargs):
    """ return r,w the return of backtest and the weights based on mode  """
    # data must be market returns.
    # weights must be a T x N dimension matrix which row i encompasses the weight of N indices of market at time i .
    if weights is None:
        weights = portfolio_stock_weights(retData, **kwargs)
        # print(weights)
    return ((retData * weights).sum(axis='columns',skipna=False), weights)


# %%
#
def cov_estimation(retSeries, mode='historic', **kwargs):
    # mode : 1- Historic 2-Elton/Gruber Constant Correlation 3-Shrinkage
    # Constant Corr : define the correlation between indices by averaging the corr based on historic data

    if mode == 'historic':
        return (retSeries.corr(),retSeries.cov()) # return (Corr, Cov)
    elif mode == 'cc':  # Constant Correlation
        stds = retSeries.std().to_numpy()
        stds = stds[np.newaxis]

        corrMat = retSeries.corr()
        n = corrMat.shape[0]
        avRho = (corrMat.sum() - n) / (n * (n - 1))
        ccMat = np.full_like(corrMat, avRho)
        np.fill_diagonal(ccMat, 1)
        return (  ccMat ,(stds.T @ stds) * ccMat)  # return (Corr, Cov)

    elif mode == 'Shrinkage':
        stds = retSeries.std().to_numpy()
        stds = stds[np.newaxis]

        corrMat = retSeries.corr()
        n = corrMat.shape[0]
        avRho = (corrMat.sum() - n) / (n * (n - 1))
        ccMat = np.full_like(corrMat, avRho)
        np.fill_diagonal(ccMat, 1)

        ShrinkageCorr= ccMat* kwargs['alpha'] +(1- kwargs['alpha'] )*retSeries.corr()
        return (  ShrinkageCorr , (stds.T @ stds) * ShrinkageCorr )  # return (Corr, Cov)


# %%
# ENC : Effective Number of Constitute exactly the same as Herfindahl index  HI= 1/ (Sigma wi**2)
# ENCB : 1/(sigma pi**2)     pi = risk contribution   Risk contribution(i)= sigma w(i)w(j)corr(i,j)/ Protfolio Var
# Naive weighting  ---> 1/N   maximize ENC / HI        Risk parity weighting ---> equal risk contribution
#  MAX ENC = EW
#  MAX ENCB = ERC
def risk_contribution(W, corrMat, stDev):
    # print("corrMat:",corrMat)
    # print("std:", stDev)
    covMat=Corr2Cov(corrMat,stDev)
    totalVariance = portfolio_expected_risk(W, stDev, corrMat) ** 2
    marginal_con  =covMat @ W

    # print("Cov:" , covMat)
    # print("Tot VAr:", totalVariance)
    # print("marginal_con:", covMat@W)
    # return np.multiply(marginal_con, W.T) / totalVariance
    return (W@covMat)*W/(totalVariance)


# %%
def contDiffs(W, targetCon, corrMat, stDev):
    # return W.sum()
    return ((risk_contribution(W, corrMat, stDev) - targetCon) ** 2).sum()


# %%
def target_risk_contribution_of_each_indice(targetRisk, corrMat, stDev):
    # print("function starts ....")
    n = corrMat.shape[0]
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda x: np.sum(x) - 1.0
                        }
    init_guess = np.repeat(1 / n, n)

    def contDiff(W, targetRisk, corrMat, stDev):
        # print(W)
        return (((risk_contribution(W, corrMat, stDev) - targetRisk) ** 2).sum())


    res = minimize(contDiffs,
                   init_guess,
                   args=(targetRisk, corrMat, stDev),
                   options={'disp': True},
                   constraints=(weights_sum_to_1,),
                   bounds=bounds,
                   method='SLSQP')
    return res.x


# %%

def equal_risk_contribution(corrMat, stDev):
    n = corrMat.shape[0]
    return target_risk_contribution_of_each_indice(np.repeat(1 / n, n), corrMat, stDev)


# %%
def regress (depRetSeries,expRetSeries,alpha=True) :
    # statmodel.api
    #You can use .summary .params   .tvalues  .pvalues .rsquared  .rsquared_adj
    import statsmodels.api as sm
    if alpha:
        expRetSeries= expRetSeries.copy()
        expRetSeries['alpha'] = 1

    return sm.OLS(depRetSeries,expRetSeries).fit()


#  %%
def portfolioTrackingError(w , bb_r , ref_r) :

    return trackingError( ref_r , (w*bb_r).sum(axis="columns") )

# %%
def styleAnalysis ( depRetSeries,expRetSeries):


    n=expRetSeries.shape[1]
    limits1={"type":'eq',
             'func': lambda x : x.sum()-1}

    mm=minimize(portfolioTrackingError,
             np.repeat(1/n,n) ,
             args=(expRetSeries,depRetSeries),
             constraints=(limits1,),
             bounds=((0,1),)*n,
             )

    return mm.x


# %%
def ffAnalysis (depRetSeries , factorSeries , rf=None):
    n=depRetSeries.shape[0]
    riskfree=pd.Series(np.repeat(rf,n))
    riskfree.index=depRetSeries.index

    return regress(depRetSeries-riskfree,factorSeries.loc[depRetSeries.index],True).params


# %%
def summaryStat(dSeries):

    if isinstance(dSeries, pd.DataFrame) or isinstance(dSeries,pd.Series):
        print( "skew is : \n" , dSeries.skew() )
        print( "kurtosis is: \n" , dSeries.kurtosis())
        return dSeries.describe()

    return "Not Correctly Formatted"





# %%

#  BLACK LITTERMAN PART COMPLETELY COPIED FROM hec

def implied_returns(delta, sigma, w):
    """
Obtain the implied expected returns by reverse engineering the weights
Inputs:
delta: Risk Aversion Coefficient (scalar)
sigma: Variance-Covariance Matrix (N x N) as DataFrame
    w: Portfolio weights (N x 1) as Series
Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir


def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)




def bl(w_prior, sigma_prior, p, q, omega=None, delta=2.5, tau=.02):
    """
    # Computes the posterior expected returns based on
    # the original black litterman reference model
    #
    # W.prior must be an N x 1 vector of weights, a Series
    # Sigma.prior is an N x N covariance matrix, a DataFrame
    # P must be a K x N matrix linking Q and the Assets, a DataFrame
    # Q must be an K x 1 vector of views, a Series
    # Omega must be a K x K matrix a DataFrame, or None
    # if Omega is None, we assume it is
    # proportional to variance of the prior
    # delta and tau are scalars
    """
    from numpy.linalg import inv
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
    #     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)


# for convenience and readability, define the inverse of a dataframe
def inverse(d):
    from numpy.linalg import inv
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)


def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w



# %%

def projJoinData(retSeries,valSeries,quotedSpreadSeries,amihudSeries):
    jointData=retSeries.join(valSeries).join(quotedSpreadSeries).join(amihudSeries)
    #print(jointData)
    colNames = []
    listedData=[]
    ctr=0
    for i in retSeries.columns:
        colNames.append((i, "Value-" + re.split("Return-", i)[1]  ,"Amihud-" + re.split("Return-", i)[1],"Qouted Spread-" + re.split("Return-", i)[1]))
        #print(tt)
        #listedData.append( jointData.loc[:, colNames[ctr]] )
        listedData.append(  [ re.split("Return-", i)[1] ,jointData.loc[:,colNames[ctr]] ] )
        listedData[ctr][1].columns=['returns','trades','amihuds','quotedspreads']
        ctr+=1
    return listedData,jointData,colNames


# %%
def plotJointAndMarginalChart(dataframe,dataName='') :
    """
       Plots joint and marginal histograms for return series and trade values.

       Args:
           dataframe (pd.DataFrame): A DataFrame with columns 'returns' and 'trades'.
       """
    # Extract return series and trade values
    returns = dataframe['returns']
    trades = dataframe['trades']

    # Create a joint histogram
    sns.jointplot(x=returns, y=trades, kind='hist', color='skyblue')
    plt.title('Joint Histogram: Returns vs. Trades | '+dataName)
    plt.xlabel('Returns')
    plt.ylabel('Trades')

    # Create marginal histograms
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(returns, color='salmon', bins=20)
    plt.title('Marginal Histogram: Returns')
    plt.xlabel('Returns')

    plt.subplot(1, 2, 2)
    sns.histplot(trades, color='green', bins=20)
    plt.title('Marginal Histogram: Trades')
    plt.xlabel('Trades')

    plt.tight_layout()
    plt.show()

# %%
def plotDensityAndScatterChart(data,dataName=''):

    g = sns.PairGrid(data, diag_sharey=False)
    #g.title('Density vs Scatter | ' + dataName)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)


# %%
def convFreqFinalData (data ,period='M'):
    ret=convertFreq(data['returns'], func=cmpt_return, period=period)
    tr=convertFreq(data['trades'], func=sum, period=period)
    qs=convertFreq(data['quotedspreads'], func=pd.DataFrame.mean, period=period)
    ami=ret/tr
    output=pd.concat([ret,tr,ami,qs],axis=1)
    output.columns=['returns','trades','amihuds','quotedspreads']
    return output

