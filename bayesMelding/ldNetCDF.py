import netCDF4 as nc4
import numpy as np
from matplotlib import pyplot as plt 
# plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from matplotlib.collections import LineCollection
from itertools import chain
import pandas as pd
import pickle
import argparse
import fnmatch
import os
import countries

# Found the logitudes transformed by Basemap are rotpole['longitude'] larger than those obtained from the netCDF files (also from R)
def rCoords(coords, pole = 193):
    coords = np.array(coords)
    coords[0] = coords[0] - pole
    coords = tuple(coords)
    return coords

def gen_img_25_storms(SEED=260, num_hatZs=200, num_tildZs = 150, flagforLdAllStorms = False):
    np.random.seed(SEED)
    plt.figure()
    mp1= Basemap(projection='rotpole',lon_0=13,o_lon_p=193,o_lat_p=41,\
           llcrnrlat = 30, urcrnrlat = 65,\
           llcrnrlon = -10, urcrnrlon = 65,resolution='c')
    mp1.drawcoastlines()
    mp1.drawcountries()
    res1 = vars(mp1)

    plt.close()

    plt.figure()
    mp = Basemap(projection='cyl',llcrnrlat=30,urcrnrlat=88,\
        llcrnrlon=-50,urcrnrlon=85,resolution='c')
    # res = m.__dict__   # This is one way to list all fields of a class
    mp.drawcoastlines()
    mp.drawcountries()

    plt.close()

    analysis_files = fnmatch.filter(os.listdir('../realData/theo/'), '*analysis*nc')
    obs_files = np.array(fnmatch.filter(os.listdir('../realData/theo_obs/'), '*obs*nc'))
    substr_obs_files = np.array([obs_files[i][:17] for i in range(len(obs_files))])
   
    for i in range(len(analysis_files)):
        analysis_file = analysis_files[i]
        obs_file = obs_files[substr_obs_files == analysis_file[:17]]

        analysis_nc = nc4.Dataset('../realData/theo/' + analysis_file, 'r')
        obs_nc = nc4.Dataset('../realData/theo_obs/' + obs_file[0], 'r')

        lon = analysis_nc.variables['grid_longitude'][:] - 360

        lat = analysis_nc.variables['grid_latitude'][:]
        data = analysis_nc.variables['max_wind_gust'][:]
       
        # The following two lines are the tricks for plt.pcolormesh
        data = data[:, :-1, :-1]

        z = np.array(data).reshape(data.shape[1],data.shape[2])
        z_min, z_max = - np.abs(z).max(), np.abs(z).max()

        obs_lon = obs_nc['longitude'][:]
        obs_lat = obs_nc['latitude'][:]
        obs_x, obs_y = mp1(obs_lon, obs_lat)
        obs_x = obs_x - 193
        obs = obs_nc['max_wind_gust'][:]
        all_X_hatZs = np.array([obs_x, obs_y]).T
        

        idx = np.random.randint(0, len(obs), num_hatZs)
        X_hatZs = all_X_hatZs[idx, :] 
        y_hatZs = obs[idx]

        # convert the lat/lon values to x/y projections.
        x, y = mp(*np.meshgrid(lon,lat))

        areal_res = 7
        res_lon = x.shape[1]/areal_res
        res_lat = x.shape[0]/areal_res

        areal_coordinate = []
        for i in range(res_lat):
            tmp0 = [np.vstack([x[i*areal_res:(i+1)*areal_res, j*areal_res:(j+1)*areal_res].ravel(), y[i*areal_res:(i+1)*areal_res, j*areal_res:(j+1)*areal_res].ravel()]).T for j in range(res_lon)]
            areal_coordinate.append(tmp0)
        areal_coordinate = np.array(list(chain.from_iterable(areal_coordinate)))
       

        areal_z = []
        for i in range(res_lat):
            tmp = [z[i*areal_res:(i+1)*areal_res, j*areal_res:(j+1)*areal_res] for j in range(res_lon)]
            areal_z.append(tmp)
        areal_z = np.array(list(chain.from_iterable(areal_z)))
        all_y_tildZs = np.array([np.mean(areal_z[i]) for i in range(len(areal_z))])
      

        idx_sampleTildZs = np.random.randint(0, len(all_y_tildZs), num_tildZs)
        y_tildZs = all_y_tildZs[idx_sampleTildZs]
        X_tildZs = areal_coordinate[idx_sampleTildZs]


        # plot the field using the fast pcolormesh routine 
        # set the colormap to jet.
        norm = mpl.colors.Normalize(vmin=0, vmax=z_max)

        plt.figure()
        # plt.pcolormesh(x,y,z, shading='flat',cmap=plt.cm.jet,  vmin=0, vmax=z_max)
        plt.pcolormesh(x,y,z, shading='flat', cmap=plt.cm.jet,  norm = norm)
        plt.colorbar()
        ax = plt.gca()

        coastsegs = res1['coastsegs']
        coastsegs = [map(rCoords, coastsegs[i]) for i in range(len(coastsegs))]
        coastlines = LineCollection(coastsegs, antialiaseds=(1,), colors = 'black')
        ax.add_collection(coastlines)

        cntrysegs, _ = mp1._readboundarydata('countries')
        cntrysegs = [map(rCoords, cntrysegs[i]) for i in range(len(cntrysegs))]
        countries = LineCollection(cntrysegs,antialiaseds=(1,), colors = 'black')
        ax.add_collection(countries)
        l, r, b, t = plt.axis() 
        # print '(l,r, b, t) is ' + str((l,r, b, t))
        plt.scatter(obs_x, obs_y, c = obs, cmap=plt.cm.jet, vmin=0, vmax=z_max,  edgecolors='black')
        plt.xlim(l, r)
        plt.ylim(b, t)
        plt.savefig(analysis_file[:17] +'.png')
        plt.show()
        plt.close()

        output_folder ='sampRealData/'+ analysis_file[:17] 
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_folder += '/'
        X_hatZs_out = open(output_folder + 'X_hatZs_seed' + str(SEED) + '.pkl', 'wb')
        pickle.dump(X_hatZs, X_hatZs_out) 
        X_hatZs_out.close()
        y_hatZs_out = open(output_folder + 'y_hatZs_seed' + str(SEED) + '.pkl', 'wb')
        pickle.dump(y_hatZs, y_hatZs_out) 
        y_hatZs_out.close()
        X_tildZs_out = open(output_folder + 'X_tildZs_seed' + str(SEED) + '.pkl', 'wb')
        pickle.dump(X_tildZs, X_tildZs_out) 
        X_tildZs_out.close()
        y_tildZs_out = open(output_folder + 'y_tildZs_seed' + str(SEED) + '.pkl', 'wb')
        pickle.dump(y_tildZs, y_tildZs_out) 
        y_tildZs_out.close()

    return [X_hatZs, y_hatZs, X_tildZs, y_tildZs]

def loadNetCdf(SEED=260, num_hatZs=200, num_tildZs = 150, crossValFlag=True, fold=10, cntry = 'FR', flagUseAvgMo = False):
    np.random.seed(SEED)
    plt.figure()
    mp1= Basemap(projection='rotpole',lon_0=13,o_lon_p=193,o_lat_p=41,\
           llcrnrlat = 30, urcrnrlat = 65,\
           llcrnrlon = -10, urcrnrlon = 65,resolution='c')
    mp1.drawcoastlines()
    mp1.drawcountries()
    res1 = vars(mp1)

    plt.close()

    plt.figure()
    mp = Basemap(projection='cyl',llcrnrlat=30,urcrnrlat=88,\
        llcrnrlon=-50,urcrnrlon=85,resolution='c')
    # res = m.__dict__   # This is one way to list all fields of a class
    mp.drawcoastlines()
    mp.drawcountries()

    # plt.close()

    analysis_files = fnmatch.filter(os.listdir('../realData/theo/'), '*analysis*nc')
    obs_files = np.array(fnmatch.filter(os.listdir('../realData/theo_obs/'), '*obs*nc'))
    substr_analysis_files = np.array([analysis_files[i][:17] for i in range(len(analysis_files))])
    substr_obs_files = np.array([obs_files[i][:17] for i in range(len(obs_files))])
    id_Imogen = np.where(substr_analysis_files == 'FPstart2016020612')[0][0]

    # for i in range(len(analysis_files)):
    i = id_Imogen
    analysis_file = analysis_files[i]
    obs_file = obs_files[substr_obs_files == analysis_file[:17]]

    analysis_nc = nc4.Dataset('../realData/theo/' + analysis_file, 'r')
    obs_nc = nc4.Dataset('../realData/theo_obs/' + obs_file[0], 'r')

    lon = analysis_nc.variables['grid_longitude'][:] - 360

    lat = analysis_nc.variables['grid_latitude'][:]
    data0 = analysis_nc.variables['max_wind_gust'][:]
   
    # The following two lines are the tricks for plt.pcolormesh
    data = data0[:, :-1, :-1]

    z = np.array(data).reshape(data.shape[1],data.shape[2])
    z_min, z_max = - np.abs(z).max(), np.abs(z).max()

    obs_lon = obs_nc['longitude'][:]
    obs_lat = obs_nc['latitude'][:]
    obs_x, obs_y = mp1(obs_lon, obs_lat)
    obs_x = obs_x - 193
    obs = obs_nc['max_wind_gust'][:]
    all_X_hatZs = np.array([obs_x, obs_y]).T
    

    idx = np.random.randint(0, len(obs), num_hatZs)
    X_hatZs = all_X_hatZs[idx, :] 
    y_hatZs = obs[idx]

    # convert the lat/lon values to x/y projections.
    x, y = mp(*np.meshgrid(lon,lat))

    if flagUseAvgMo:
        areal_res = 7
        res_lon = x.shape[1]/areal_res
        res_lat = x.shape[0]/areal_res

        areal_coordinate = []
        for i in range(res_lat):
            tmp0 = [np.vstack([x[i*areal_res:(i+1)*areal_res, j*areal_res:(j+1)*areal_res].ravel(), y[i*areal_res:(i+1)*areal_res, j*areal_res:(j+1)*areal_res].ravel()]).T for j in range(res_lon)]
            areal_coordinate.append(tmp0)
        areal_coordinate = np.array(list(chain.from_iterable(areal_coordinate)))
       

        areal_z = []
        for i in range(res_lat):
            tmp = [z[i*areal_res:(i+1)*areal_res, j*areal_res:(j+1)*areal_res] for j in range(res_lon)]
            areal_z.append(tmp)
        areal_z = np.array(list(chain.from_iterable(areal_z)))
        all_y_tildZs = np.array([np.mean(areal_z[i]) for i in range(len(areal_z))])
      

        idx_sampleTildZs = np.random.randint(0, len(all_y_tildZs), num_tildZs)
        y_tildZs = all_y_tildZs[idx_sampleTildZs]
        X_tildZs = areal_coordinate[idx_sampleTildZs]


        # plot the field using the fast pcolormesh routine 
        # set the colormap to jet.
        norm = mpl.colors.Normalize(vmin=0, vmax=z_max)

        plt.figure()
        # plt.pcolormesh(x,y,z, shading='flat',cmap=plt.cm.jet,  vmin=0, vmax=z_max)
        plt.pcolormesh(x,y,z, shading='flat', cmap=plt.cm.jet,  norm = norm)
        plt.colorbar()
        ax = plt.gca()

        coastsegs = res1['coastsegs']
        coastsegs = [map(rCoords, coastsegs[i]) for i in range(len(coastsegs))]
        coastlines = LineCollection(coastsegs, antialiaseds=(1,), colors = 'black')
        ax.add_collection(coastlines)

        cntrysegs, _ = mp1._readboundarydata('countries')
        cntrysegs = [map(rCoords, cntrysegs[i]) for i in range(len(cntrysegs))]
        countries = LineCollection(cntrysegs,antialiaseds=(1,), colors = 'black')
        ax.add_collection(countries)
        l, r, b, t = plt.axis() 
        # print '(l,r, b, t) is ' + str((l,r, b, t))
        plt.scatter(obs_x, obs_y, c = obs, cmap=plt.cm.jet, vmin=0, vmax=z_max,  edgecolors='black')
        plt.xlim(l, r)
        plt.ylim(b, t)
        plt.savefig(analysis_file[:17] + '_' + str(cntry) +'.png')
        plt.show()
        plt.close()

        output_folder ='sampRealData/'+ analysis_file[:17] + '_' + str(cntry) + '_numObs_' + str(num_hatZs) + '_numMo_' + str(num_tildZs) 
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_folder += '/'
        X_hatZs_out = open(output_folder + 'X_hatZs_seed' + str(SEED)  + '.pkl', 'wb s')
        pickle.dump(X_hatZs, X_hatZs_out) 
        X_hatZs_out.close()
        y_hatZs_out = open(output_folder + 'y_hatZs_seed' + str(SEED) + '.pkl', 'wb')
        pickle.dump(y_hatZs, y_hatZs_out) 
        y_hatZs_out.close()
        X_tildZs_out = open(output_folder + 'X_tildZs_seed' + str(SEED)  + '.pkl', 'wb')
        pickle.dump(X_tildZs, X_tildZs_out) 
        X_tildZs_out.close()
        y_tildZs_out = open(output_folder + 'y_tildZs_seed' + str(SEED)  + '.pkl', 'wb')
        pickle.dump(y_tildZs, y_tildZs_out) 
        y_tildZs_out.close()
        return [X_hatZs, y_hatZs, X_tildZs, y_tildZs]
    else:
        lon_Mo = x[:, :].ravel()
        lat_Mo = y[:, :].ravel()
        data_Mo = np.array([lon_Mo, lat_Mo, data0.ravel()]).T
        data_Obs = np.array([obs_x, obs_y, obs]).T
        # cntryNames = [find_cntryNames(coords_Mo[i]) for i in range(len(coords_Mo))] # due to the large \
        #size of coordinates (around 800,000), find all coordinates that were within the square that includes France 
        dataMo_within_squareAround_FR = coords_within_squareAround_FR(data_Mo)
        dataObs_within_squareAround_FR = coords_within_squareAround_FR(data_Obs)
        # print (dataMo_within_squareAround_FR.shape, dataObs_within_squareAround_FR.shape, dataObs_within_squareAround_FR[:10, 2])
        # exit(-1)

        # plot the field using the fast pcolormesh routine 
        # set the colormap to jet.
        norm = mpl.colors.Normalize(vmin=0, vmax=z_max)

        plt.figure()
        # plt.pcolormesh(x,y,z, shading='flat',cmap=plt.cm.jet,  vmin=0, vmax=z_max)
        plt.pcolormesh(x,y,z, shading='flat', cmap=plt.cm.jet,  norm = norm)
        plt.colorbar()
        ax = plt.gca()

        coastsegs = res1['coastsegs']
        coastsegs = [map(rCoords, coastsegs[i]) for i in range(len(coastsegs))]
        coastlines = LineCollection(coastsegs, antialiaseds=(1,), colors = 'black')
        ax.add_collection(coastlines)

        cntrysegs, _ = mp1._readboundarydata('countries')
        cntrysegs = [map(rCoords, cntrysegs[i]) for i in range(len(cntrysegs))]
        countries = LineCollection(cntrysegs,antialiaseds=(1,), colors = 'black')
        ax.add_collection(countries)
        l, r, b, t = plt.axis() 

        plt.scatter(dataObs_within_squareAround_FR[:, 0], dataObs_within_squareAround_FR[:, 1], \
            c = dataObs_within_squareAround_FR[:, 2], cmap=plt.cm.jet, vmin=0, vmax=z_max, edgecolors='black')
        plt.xlim(l, r)
        plt.ylim(b, t)
        plt.savefig(analysis_file[:17] + '_' + str(cntry) +'.png')
        plt.show()
        plt.close()
        idx = np.random.randint(0, dataObs_within_squareAround_FR.shape[0], num_hatZs)
        X_hatZs = dataObs_within_squareAround_FR[idx, :2]
        y_hatZs = dataObs_within_squareAround_FR[idx, 2]
        #remove the mean of the observations
        y_hatZs = y_hatZs - np.mean(y_hatZs)

        idx_sampleTildZs = np.random.randint(0, dataMo_within_squareAround_FR.shape[0], num_tildZs)
        X_tildZs_tmp = dataMo_within_squareAround_FR[idx_sampleTildZs, :2]

        X_tildZs = np.array([X_tildZs_tmp[i].reshape(1, X_tildZs_tmp.shape[1]) for i in range(X_tildZs_tmp.shape[0])])
        y_tildZs = dataMo_within_squareAround_FR[idx_sampleTildZs, 2]

        areal_hatZs = []
        for i in range(len(y_tildZs)):
            idx_min_dist = np.argmin(np.array([np.linalg.norm(X_tildZs[i] - X_hatZs[j]) for j in range(len(y_hatZs))]))
            areal_hatZs.append(y_hatZs[idx_min_dist])
        areal_hatZs = np.array(areal_hatZs)

        output_folder ='sampRealData/'+ analysis_file[:17] + '_' + str(cntry) + '_numObs_' + str(num_hatZs) + \
        '_numMo_' + str(num_tildZs) +  '/seed' + str(SEED)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_folder += '/'
        num_hatZs = X_hatZs.shape[0]
        if crossValFlag:
            mask0 = np.ones(num_hatZs, dtype=bool)
            mask_idx = np.arange(X_hatZs.shape[0])
            num_per_fold = num_hatZs/fold
            for i in range(fold):
                include_idx = mask_idx[i * num_per_fold:(i + 1) * num_per_fold]
                mask = np.copy(mask0)
                mask[include_idx] = False
                X_train = X_hatZs[mask, :] 
                y_train = y_hatZs[mask]
                X_test = X_hatZs[~mask, :]
                y_test = y_hatZs[~mask]
                if i == fold -1:
                    include_idx = mask_idx[i * num_per_fold:(i + 1) * num_per_fold + num_hatZs % fold]
                    mask = np.copy(mask0)
                    mask[include_idx] = False
                    X_train = X_hatZs[mask, :] 
                    y_train = y_hatZs[mask]
                    X_test = X_hatZs[~mask, :]
                    y_test = y_hatZs[~mask]
                output_folder_cv = output_folder + 'fold_' + str(i)
                if not os.path.exists(output_folder_cv):
                    os.makedirs(output_folder_cv)
                output_folder_cv += '/'
                X_train_out = open(output_folder_cv + 'X_train.pkl', 'wb')
                pickle.dump(X_train, X_train_out)
                y_train_out = open(output_folder_cv + 'y_train.pkl', 'wb')
                pickle.dump(y_train, y_train_out)
                X_test_out = open(output_folder_cv  + 'X_test.pkl', 'wb')
                pickle.dump(X_test, X_test_out)
                y_test_out = open(output_folder_cv + 'y_test.pkl', 'wb')
                pickle.dump(y_test, y_test_out)
                X_tildZs_out = open(output_folder_cv + 'X_tildZs.pkl', 'wb')
                pickle.dump(X_tildZs, X_tildZs_out) 
                X_tildZs_out.close()
                y_tildZs_out = open(output_folder_cv + 'y_tildZs.pkl', 'wb')
                pickle.dump(y_tildZs, y_tildZs_out) 
                y_tildZs_out.close()
            return [X_train, y_train, X_test, y_test, X_tildZs, y_tildZs]
        else:
            X_hatZs_out = open(output_folder + 'X_hatZs.pkl', 'wb')
            pickle.dump(X_hatZs, X_hatZs_out) 
            X_hatZs_out.close()
            y_hatZs_out = open(output_folder + 'y_hatZs.pkl', 'wb')
            pickle.dump(y_hatZs, y_hatZs_out) 
            y_hatZs_out.close()
            X_tildZs_out = open(output_folder + 'X_tildZs.pkl', 'wb')
            pickle.dump(X_tildZs, X_tildZs_out) 
            X_tildZs_out.close()
            y_tildZs_out = open(output_folder + 'y_tildZs.pkl', 'wb')
            pickle.dump(y_tildZs, y_tildZs_out) 
            y_tildZs_out.close()
            areal_hatZs_out = open(output_folder + 'areal_hatZs.pkl', 'wb')
            pickle.dump(areal_hatZs, areal_hatZs_out) 
            y_tildZs_out.close()
            return [X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_hatZs]

# The following code translated from R is NOT working for Python Basemap.
# def rotateCoords(coords, pole = [193, 41]):
#     if len(coords) == 2:
#         coords = np.array(coords).reshape(1,len(coords))
#     degtorad = np.pi / 180.
#     pole_long = (pole[0] % 360.) * degtorad
#     pole_latt = (pole[1] % 360.) * degtorad
#     SOCK = pole_long - np.pi
#     if pole_long ==0:
#         SOCK = 0
#     longit = coords[:, 0] % 360. * degtorad
#     latt  = coords[:, 1] % 360. * degtorad
#     SCREW  = longit - SOCK
#     SCREW = SCREW % (2 * np.pi)
#     BPART = np.cos(SCREW) * np.cos(latt)
#     rlatt = np.arcsin(-np.cos(pole_latt) * BPART + np.sin(pole_latt) * np.sin(latt))
#     t1 = np.cos(pole_latt) * np.sin(latt)
#     t2 = np.sin(pole_latt) * BPART
#     rlong = -np.arccos((t1 + t2) / np.cos(rlatt))
#     ind = np.logical_or(np.logical_and(0 < SCREW, SCREW < np.pi), SCREW > np.pi)
#     rlong[ind] = -rlong[ind]
#     rCoords = 180. * np.array([rlong, rlatt])/np.pi
#     rCoords = rCoords.T
#     return rCoords
def find_cntryNames(coords):
    try:
        res = get_country(coords)
    except:
        res = 'not_cntry'
    return res
def coords_within_squareAround_FR(X):
    xmin = -12.5
    xmax = -2.5
    ymin = -6.5
    ymax = 2.5
    data_within_squareAround_FR = X[(X[:, 0]>=xmin) & (X[:, 0]<=xmax) & (X[:,1]>=ymin) & (X[:,1]<=ymax)]
    return data_within_squareAround_FR


def get_country(coords):
    cc = countries.CountryChecker('./TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp')
    res = cc.getCountry(countries.Point(coords[0], coords[1])).iso
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=99, help='The simulation index')
    p.add_argument('-numObs', type=int, dest='numObs', default=328, help='Number of observations used in modelling')
    p.add_argument('-numMo', type=int, dest='numMo', default=300, help='Number of model outputs used in modelling')
    p.add_argument('-crossValFlag', dest='crossValFlag', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='whether to validate the model using cross validation')
    args = p.parse_args()
    loadNetCdf(args.SEED, args.numObs, args.numMo, args.crossValFlag)



    



