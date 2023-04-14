__author__ = 'German Perez Fogwill'

import pandas as pd
import datetime
import numpy as np
import os
from optparse import OptionParser
import math     # import math library
import scipy    # import scientific python functions


VERSIONSTR = "\n%prog v. 0.1 2015 byNegro\n"
INPUTDIR = 'C:\\Users\\Administrador\\Desktop\\la'  # os.getcwd()
OUTPUTDIR = os.getcwd()

input_file = 'lqi2016.dat'
header_file = 'Headline_1997.dat'

# %%function
def _arraytest(*args):
    """
    Function to convert input parameters in as lists or tuples to
    arrays, while leaving single values intact.

    Test function for single values or valid array parameter input
    (J. Delsman).

    Parameters:
        args (array, list, tuple, int, float): Input values for functions.

    Returns:
        rargs (array, int, float): Valid single value or array function input.

    Examples
    --------

        >>> _arraytest(12.76)
        12.76
        >>> _arraytest([(1,2,3,4,5),(6,7,8,9)])
        array([(1, 2, 3, 4, 5), (6, 7, 8, 9)], dtype=object)
        >>> x=[1.2,3.6,0.8,1.7]
        >>> _arraytest(x)
        array([ 1.2,  3.6,  0.8,  1.7])
        >>> _arraytest('This is a string')
        'This is a string'

    """

    rargs=[]
    for a in args:
        if isinstance(a, (list, tuple)):
            rargs.append(scipy.array(a))
        else:
            rargs.append(a)
    if len(rargs) == 1:
        return rargs[0] # no unpacking if single value, return value i/o list
    else:
        return rargs


def windvec(aux_wind):
    """
    Function to calculate the wind vector from time series of wind
    speed and direction.

    Parameters:
        - u: array of wind speeds [m s-1].
        - D: array of wind directions [degrees from North].

    Returns:
        - uv: Vector wind speed [m s-1].
        - Dv: Vector wind direction [degrees from North].

    Examples
    --------

        >>> u = scipy.array([[ 3.],[7.5],[2.1]])
        >>> D = scipy.array([[340],[356],[2]])
        >>> windvec(u,D)
        (4.162354202836905, array([ 353.2118882]))
        >>> uv, Dv = windvec(u,D)
        >>> uv
        4.162354202836905
        >>> Dv
        array([ 353.2118882])

    """


    aux_wind = aux_wind.str.split('_')

    u = aux_wind.apply(lambda x: x[0])
    D = aux_wind.apply(lambda x: x[1])

    u = u.values
    D = D.values

    u = [float(i) for i in u]
    D = [float(i) for i in D]

    if u.__len__() == 0:
        return np.nan, np.nan

    # Test input array/value
    u, D = _arraytest(u,D)

    ve = 0.0 # define east component of wind speed
    vn = 0.0 # define north component of wind speed
    D = D * math.pi / 180.0 # convert wind direction degrees to radians
    for i in range(0, len(u)):
        ve = ve + u[i] * math.sin(D[i]) # calculate sum east speed components
        vn = vn + u[i] * math.cos(D[i]) # calculate sum north speed components
    ve = - ve / len(u) # determine average east speed component
    vn = - vn / len(u) # determine average north speed component
    uv = math.sqrt(ve * ve + vn * vn) # calculate wind speed vector magnitude
    # Calculate wind speed vector direction
    vdir = scipy.arctan2(ve, vn)
    vdir = vdir * 180.0 / math.pi # Convert radians to degrees
    if vdir < 180:
        Dv = vdir + 180.0
    else:
        if vdir > 180.0:
            Dv = vdir - 180
        else:
            Dv = vdir
    return uv, Dv # uv in m/s, Dv in dgerees from North
# %% end of function

def load_new_files(input_dir, year):

    ozn_files_list = []
    met_files_list = []
    header_file_list = []
    df = pd.DataFrame()
    df_met = pd.DataFrame()

    for file in os.listdir(input_dir):
        if file.endswith(".txt") | file.endswith('.TXT'):
            header_file_list.append(file)

    # Listo los headers para ver si hay mas de uno.
    for file in os.listdir(input_dir):
        if file.endswith(".txt") | file.endswith('.TXT'):
            header_file_list.append(file)
        elif file.endswith(".ozn") | file.endswith('.OZN'):
            # ozn_files_list.append(file)
            df = df.append(load_TEI49C_file(input_dir, file, year))
        elif file.endswith(".met") | file.endswith('.MET'):
            # met_files_list.append(file)
            if header_file_list.__len__() != 1:
                i = 0
                for p in header_file_list:
                    print('[' + str(i) + ']:\t' + p)
                    i += 1
                var = input("Seleccione el archivo del HEADER para procesar el archivo " + file)
                aux_df = load_cr10_file(file, header_file_list[int(var)], year)
                df_met = df_met.append(aux_df)

            else:
                aux_df = load_lqo_2014_met(file, header_file_list[0], year)
                #aux_df = load_cr10_file(file, header_file_list[0], year)
                df_met = df_met.append(aux_df)

    return df, df_met


def load_lqo_2014_met(input_file, header_file, year):
    met = pd.read_csv(INPUTDIR + '\\2014LQO.met', parse_dates=[[0, 1]], dayfirst =True)
    met_df = pd.DataFrame(index=met.FECHA_HORA)

    met_df['WD'] = met.WD.data
    met_df['WS'] = met.WS.data
    met_df['RH'] = met.RH.data
    met_df['AP'] = met.AP.data
    met_df['AT'] = met.AT.data

    met_df['WD'] = met_df['WD'] * 10
    met_df['WS'] = met_df['WS'] * 0.514444
    met_df.loc[met_df.WD == 0, ['WD']] = 360

    return met_df


def load_mbi_file(input_dir, year):

    df_data = pd.DataFrame.from_csv(input_dir + '\\' + 'mbiomet' + year + '.DAT', sep=',', infer_datetime_format=True)
    df_met = pd.DataFrame.from_csv(input_dir + '\\' + 'mbiomet' + year + '.DAT', sep=',', infer_datetime_format=True)

    df_data.columns = ['WD', 'WS', 'AT', 'RH', 'DT', 'AP', 'NA', 'NA', 'NA', 'NA', 'O3']
    df_met.columns = ['WD', 'WS', 'AT', 'RH', 'DT', 'AP', 'NA', 'NA', 'NA', 'NA', 'O3']

    return df_data, df_met


def load_TEI49C_file(input_dir, file, year):

    df = []
    full_dir = input_dir + '\\' + file

    fi = pd.read_csv(full_dir, delim_whitespace=True, header=5, comment=';', parse_dates=[[0,1]], error_bad_lines=False, warn_bad_lines=False)
    fi.columns = ['Time_Date', 'Alarms', 'O3', 'IntenA', 'IntenB', 'Bench_Temp', 'LampTemp', 'O3LampTemp', 'FlowA', 'FlowB', 'Pres']

    data_time = pd.DatetimeIndex(fi['Time_Date'])
    data_time = data_time + pd.DateOffset(year=int(year))

    data = np.array(fi.ix[:, 2::], dtype=np.float64)
    df = pd.DataFrame(data=data, index=data_time)

    df.columns = ['O3', 'IntenA', 'IntenB', 'Bench_Temp', 'LampTemp', 'O3LampTemp', 'FlowA', 'FlowB', 'Pres']

    return df


def load_old_files(input_dir, year):

    files_list = []
    header_file_list = []
    df = pd.DataFrame()

    # Listo los headers para ver si hay mas de uno.
    for file in os.listdir(input_dir):
        if file.endswith(".txt") | file.endswith('.TXT'):
            header_file_list.append(file)

    # Listo los archivos que voy a cargar en memoria
    for file in os.listdir(input_dir):
        if file.endswith(".dat") | file.endswith('.DAT'):

            if (header_file_list.__len__() != 1):
                i = 0
                for p in header_file_list:
                    print('[' + str(i) + ']:\t' + p)
                    i += 1
                var = input("Seleccione el archivo del HEADER para procesar el archivo:\t " + file)
                year = int(input("Seleccione el año del archivo procesar el archivo:\t "))
                aux_df = load_cr10_file(file, header_file_list[int(var)], year)
                df = df.append(aux_df[['WD', 'WS', 'RH', 'AP', 'AT', 'O3']])

            else:
                aux_df = load_cr10_file(file, header_file_list, year)
                df = df.append(aux_df)

    return df, df


def load_cr10_file(input_file, header_file, year):

    #with open(INPUTDIR + '\\' + header_file) as f:
    with open(header_file) as f:
        header = f.readlines()

    header = header[0].split(',')

    #file = pd.read_csv(INPUTDIR + '\\' + input_file, error_bad_lines=False, warn_bad_lines=False)
    file = pd.read_csv(input_file, error_bad_lines=False, warn_bad_lines=False)

    file = file.ix[:, 0:header.__len__()]
    file = file[file.ix[:, 0] != 102]
    file.columns = header

    time_str = [str(i) for i in file['TIME (UTC)']]
    aux = []
    for i in time_str:
        aux.append('0'*(4-i.__len__())+i)

    # time_str = aux

    b = ",".join(str(int(i)) for i in file.ix[:, 'TIME (UTC)'])
    b = b.split(',')
    b = [i.rjust(4, '0') for i in b]
    file.ix[:, 'TIME (UTC)'] = b

    a = (file['Julian Day'].map(int)).map(str) + ' ' + file['TIME (UTC)']

    time = []

    for row in a:

        try:
            time.append(datetime.datetime.strptime(str(year) + ' ' + row, "%Y %j %H%M"))
        except ValueError:
            aux_time = row.replace(' 24', ' 23')
            aux_time = datetime.datetime.strptime(str(year) + ' ' + aux_time, "%Y %j %H%M")
            aux_time += datetime.timedelta(hours=1)
            time.append(aux_time)

    data = np.array(file.ix[:, 3::], dtype=np.float64)
    df = pd.DataFrame(data=data, index=time)
    df.columns = header[3::]

    return df


def resample_met(df_met, period_to_resample):

    aux_wind = df_met['WS'].map(str) + '_' + df_met['WD'].map(str)

    df_met = df_met.resample(period_to_resample, how=np.mean)
    aux_wind = aux_wind.resample(period_to_resample, how=windvec)
    df_met['WS'] = aux_wind.apply(lambda x: x[0])
    df_met['WD'] = aux_wind.apply(lambda x: x[1])

    return df_met


def process_data(df_data, df_met, max_std, min_ws, max_ws, max_wd, year):

    o3_data = pd.DataFrame()
    met_data = pd.DataFrame()
    df_met = resample_met(df_met, 'H')

    df_data.loc[((df_data.O3 > 100) | (df_data.O3 < 0)), ['O3']] = np.nan

    # Datos de Ozono
    o3_data['DATE'] = '9999-99-99'
    o3_data['TIME'] = '99:99'
    o3_data['O3'] = df_data.O3.resample('H', how='mean')
    o3_data['ND'] = df_data.O3.resample('H', how='count')
    o3_data['SD'] = df_data.O3.resample('H', how=np.std)
    o3_data['F'] = 0
    o3_data['CS'] = 0
    o3_data['REM'] = -99999999
    o3_data['DATE'] = '9999-99-99'
    o3_data['TIME'] = '99:99'

    o3_data.O3 = o3_data.O3.round(1)
    o3_data.SD = o3_data.SD.round(2)

    # Datos de Meteorologia
    met_data['WD'] = df_met.WD
    met_data['WS'] = df_met.WS
    met_data['RH'] = df_met.RH
    met_data['AP'] = df_met.AP
    met_data['AT'] = df_met.AT

    met_data.WD = met_data.WD.round(1)
    met_data.WS = met_data.WS.round(1)
    met_data.RH = met_data.RH.round(1) 
    met_data.AP = met_data.AP.round(1)
    met_data.AT = met_data.AT.round(1)

    df_data = df_data.resample('H', how='mean')
    df_data.WD = df_met.WD
    df_data.WS = df_met.WS

    # Completo las series de tiempo
    o3_data = o3_data.reindex(pd.date_range(pd.datetime(int(year), 1, 1, 0, 0), pd.datetime(int(year), 12, 31, 23, 59), freq='H'))
    met_data = met_data.reindex(pd.date_range(pd.datetime(int(year), 1, 1, 0, 0), pd.datetime(int(year), 12, 31, 23, 59), freq='H'))

    # Pongo las banderas
    o3_data.loc[((o3_data.SD > max_std) | (met_data.WD < max_wd) | (met_data.WS > max_ws)), 'F'] = 1

    # Pongo datos no validos
    o3_data.loc[o3_data.DATE.isnull(), ['DATE', 'TIME', 'O3', 'ND', 'SD', 'F', 'CS', 'REM']] = ['9999-99-99', '99:99', -9999999.9, -9999, -99.99, -9999, 0, -99999999]
    o3_data.loc[np.isnan(o3_data.O3), ['O3', 'ND', 'SD', 'F']] = [-9999999.9, -9999, -99.99, -9999]
    o3_data.loc[o3_data.O3 == 0, ['O3', 'ND', 'SD', 'F']] = [-9999999.9, -9999, -99.99, -9999]
    o3_data.loc[o3_data.ND > 60, ['ND']] = [60]
    o3_data.loc[np.isnan(o3_data.SD), ['SD']] = [-99.99]

    #o3_data.loc[o3_data.ND < 40, ['O3', 'ND', 'SD', 'F']] = [-9999999.9, -9999, -99.99, -9999]
    o3_data.loc[o3_data.ND < 3, ['O3', 'ND', 'SD', 'F']] = [-9999999.9, -9999, -99.99, -9999]

    o3_data.loc[((o3_data.O3 == 0) & (o3_data.SD == 0)), ['O3', 'ND', 'SD', 'F']] = [-9999999.9, -9999, -99.99, -9999]

    met_data.loc[np.isnan(met_data.RH) | np.isnan(met_data.WD), ['WD', 'WS', 'RH', 'AP', 'AT']] = [-99.9, -99.9, -99.9, -999.9, -99.9]
    met_data.loc[((met_data.WD == 0) | (np.isnan(met_data.WD))), ['WD', 'WS']] = [-99.9, -99.9]
    met_data.loc[((met_data.AT < -40) | (met_data.AT > 40)), ['AT']] = [-99.9]

    # Promedios horarios y mensuales
    o3_data_D = process_daily(o3_data, year)
    o3_data_M = process_monthly(o3_data_D, year)

    # Cambio formato de hora
    o3_data.index = o3_data.index.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    met_data.index = met_data.index.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    o3_data_D.index = o3_data_D.index.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    o3_data_M.index = o3_data_M.index.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))

    return o3_data, met_data, o3_data_D, o3_data_M


def process_daily(o3_data, year):

    o3_data_D = pd.DataFrame()

    # Datos de Ozono
    o3_data_D['DATE'] = '9999-99-99'
    o3_data_D['TIME'] = '99:99'
    o3_data_D['O3'] = o3_data.O3[o3_data.F == 0].resample('D', how='mean')
    o3_data_D['ND'] = o3_data.O3[o3_data.F == 0].resample('D', how='count')
    o3_data_D['SD'] = o3_data.O3[o3_data.F == 0].resample('D', how=np.std)
    o3_data_D['F'] = -9999
    o3_data_D['CS'] = 0
    o3_data_D['REM'] = -99999999
    o3_data_D['DATE'] = '9999-99-99'
    o3_data_D['TIME'] = '99:99'

    o3_data_D.O3 = o3_data_D.O3.round(1)
    o3_data_D.SD = o3_data_D.SD.round(2)

    o3_data_D = o3_data_D.reindex(pd.date_range(pd.datetime(int(year), 1, 1, 0, 0), pd.datetime(int(year), 12, 31, 23, 59), freq='D'))

    o3_data_D.loc[np.isnan(o3_data_D.O3), ['DATE', 'TIME', 'O3', 'ND', 'SD', 'F', 'CS', 'REM']] = ['9999-99-99', '99:99', -9999999.9, -9999, -99.99, -9999, 0, -99999999]

    return o3_data_D


def process_monthly(o3_data_D, year):

    o3_data_M = pd.DataFrame()

    # Datos de Ozono
    o3_data_M['DATE'] = '9999-99-99'
    o3_data_M['TIME'] = '99:99'
    o3_data_M['O3'] = o3_data_D.O3[o3_data_D.O3 != -9999999.9].resample('MS', how='mean')
    o3_data_M['ND'] = o3_data_D.O3[o3_data_D.O3 != -9999999.9].resample('MS', how='count')
    o3_data_M['SD'] = o3_data_D.O3[o3_data_D.O3 != -9999999.9].resample('MS', how=np.std)
    o3_data_M['F'] = -9999
    o3_data_M['CS'] = 0
    o3_data_M['REM'] = -99999999
    o3_data_M['DATE'] = '9999-99-99'
    o3_data_M['TIME'] = '99:99'

    o3_data_M.O3 = o3_data_M.O3.round(1)
    o3_data_M.SD = o3_data_M.SD.round(2)

    o3_data_M = o3_data_M.reindex(pd.date_range(pd.datetime(int(year), 1, 1, 0, 0), pd.datetime(int(year), 12, 31, 23, 59), freq='MS'))

    o3_data_M.loc[np.isnan(o3_data_M.O3), ['DATE', 'TIME', 'O3', 'ND', 'SD', 'F', 'CS', 'REM']] = ['9999-99-99', '99:99', -9999999.9, -9999, -99.99, -9999, 0, -99999999]

    o3_data_M.loc[np.isnan(o3_data_M.SD), ['SD']] = [-99.99]

    return o3_data_M


def save_data(o3_data, met_data, o3_data_D, o3_data_M):

    f = open('O3_H.dat', 'wt')
    f.write(o3_data.to_string())
    f.close()

    f = open('MET_H.dat', 'wt')
    f.write(met_data.to_string())
    f.close()

    f = open('O3_D.dat', 'wt')
    f.write(o3_data_D.to_string())
    f.close()

    f = open('O3_M.dat', 'wt')
    f.write(o3_data_M.to_string())
    f.close()

    return


def main():
      
    usagestr = "\n\t%prog [options]" \
    "\nProcesador de los datos de Ozono superficial.\n" \
    "(Ver las opciones usando: --help)"
 
    parser = OptionParser(usage=usagestr, version=VERSIONSTR)
    parser.add_option("-i", "--input", dest="inputdir", default=INPUTDIR, help="Directorio de entrada ,default:%s" %INPUTDIR)
    parser.add_option("-o", "--output", dest="outputdir", default=OUTPUTDIR, help="Directorio de salida ,default:%s" %OUTPUTDIR)
    parser.add_option("-y", "--year", dest="year", default='2016', help="Año de datos a procesar, default:")
    parser.add_option("-s", "--station", dest="station", default="lqi", help="Estacion a procesar (mbi, lqi, sju, pil), default: mbi")
    parser.add_option("-f", "--format", dest="format", default="nuevo", help="Formato de los archivos 'viejo' o 'nuevo', default: nuevo")
    (opts, args) = parser.parse_args()
    parser.print_usage()

    print("Directorio de entrada:", opts.inputdir)
    print("Directorio de salida :", opts.outputdir)
    print("Año a procesar       :", opts.year)
    print("Estacion             :", opts.station)
    print("Formato              :", opts.format)

    if opts.station == 'mbi':
        max_std = 1.75
        min_ws = 2
        max_ws = 20
        max_wd = 90
    if opts.station == 'pil':
        max_std = 2
        min_ws = 2
        max_ws = 20
        max_wd = 0
    if opts.station == 'sju':
        max_std = 2
        min_ws = 2
        max_ws = 20
        max_wd = 0
    if opts.station == 'lqi':
        max_std = 1.75
        min_ws = 2
        max_ws = 20
        max_wd = 0

    # Cargo todos los archivos en memoria
    if opts.station == 'mbi':
        df_data, df_met = load_mbi_file(opts.inputdir, opts.year)
    elif opts.format != 'nuevo':
        df_data, df_met = load_old_files(opts.inputdir, opts.year)
    else:
        df_data, df_met = load_new_files(opts.inputdir, opts.year)

    # Proceso los datos
    o3_data, met_data, o3_data_D, o3_data_M = process_data(df_data, df_met, max_std, min_ws, max_ws, max_wd, opts.year)

    # Guardo los datos
    save_data(o3_data, met_data, o3_data_D, o3_data_M)

    print('ok')

if __name__ == "__main__":    # if not a module, execute main()
    main()