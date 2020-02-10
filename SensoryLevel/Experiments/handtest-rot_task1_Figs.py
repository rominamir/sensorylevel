import json
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import colorsys
import pickle
from warnings import simplefilter
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols

import os




import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'cm'

extended_view = False
sensory_versions = 3
select_sensory = range(sensory_versions)

RL_method = "PPO1"
total_MC_runs = 55 #50 # starts from 1
experiment_ID = "experiment_4_pool_with_MC_C_task1"
total_timesteps = 100000
episode_timesteps = 1000
total_episodes = int(total_timesteps/episode_timesteps)

episode_rewards_all = np.zeros([total_MC_runs, sensory_versions, total_episodes])
# switch_MC = np.zeros([total_MC_runs, sensory_versions]) 

for sensory_value in range(sensory_versions):
	sensory_value_str = "sensory_{}".format(sensory_value)
	for mc_cntr in range(total_MC_runs):
		# log_dir = "./logs/{}/MC_{}/{}/{}/".format(experiment_ID, mc_cntr, RL_method, sensory_value_str)
		jsonFile = open("C:\\Users\\Romin\\Projects\\HandManipulation\\hand_test\\logs\\handtest_rot_pool_with_MC_C_task1\\MC_{}\\PPO1\\{}\\Monitor\\openaigym.episode_batch.0.Monitor_info.stats.json".format(mc_cntr,sensory_value_str) )
		jsonString = jsonFile.read()
		jsonData = json.loads(jsonString)
		# print("sensory_info: ", sensory_value , "mc_cntr: ", mc_cntr)
		episode_rewards_all[mc_cntr, sensory_value] = np.array(jsonData['episode_rewards'])
reward_to_displacement_coeficient  = 1
episode_displacement_all = episode_rewards_all*reward_to_displacement_coeficient
# print (np.shape(episode_rewards_all))
episode_displacement_average = episode_displacement_all.mean(0)
episode_displacement_std = episode_displacement_all.std(0)
# episode_displacement_std1 = episode_displacement_all.std(2)
Smean = episode_rewards_all.mean(2)
# print (np.shape(episode_rewards_all.mean(2)))
final_displacement = np.zeros([total_MC_runs, sensory_versions])
pass_displacement_point = np.zeros([total_MC_runs, sensory_versions])
# print (np.shape(pass_displacement_point))

displacement_point = 10


sensory_info_full = ['none', 'one SD', 'Normal Force','one SD', '3D Force', 'one SD']
sensory_info = ['No Force', '1D Force','3D Force']
'''
# if extended_view:
	# sensory_values = sensory_values_full
	# senory_values_legend = sensory_values_legend_full
# else:
	# sesnory_values = ["0", "1k", "4k", "10k", "20k"]
	# sensory_values_legend = ["K: 0", "K: 1k", "K: 4k", "K: 10k", "K: 20k"]
'''
'''
for sesnory_value in range(sensory_versions):
	for mc_cntr in range(total_MC_runs):
		if episode_displacement_std1[mc_cntr][sesnory_value] >= 5 :
			switch_MC[mc_cntr][sensory_value] == 1
		else:
			switch_MC[mc_cntr][sesnory_value]==0
print(switch_MC)
'''


## Figure 1
fig = plt.figure(figsize= (7,4))

for sensory_value in select_sensory:
	x0=range(total_episodes)
	y0=episode_displacement_average[sensory_value, :]
	std0 = episode_displacement_std[sensory_value,:]
	# plt.subplot(3,1,sensory_value+1)
	plt.plot(x0, y0, alpha=.75)
	# plt.fill_between(x0, y0-std0/2, y0+std0/2, color=colorsys.hsv_to_rgb((8.75-sensory_value)/14,1,.75), alpha=0.20)
	# plt.yticks(np.arange(0, 210,50), fontsize=8)

	# fig.legend(sensory_info_full, fontsize='x-small',loc='upper right')
	plt.legend(sensory_info, fontsize='small',loc='upper left')


# plt.xlabel('Episode #', fontsize=8)
# plt.ylabel('Mean Reward', fontsize=8)
# plt.xticks(np.arange(0,4001,50),rotation=45, fontsize=8)

fig.text(0.5, 0.04, 'Episode #', ha='center', fontsize=8)
fig.text(0.05, 0.5, 'Mean Reward', va='center', rotation='vertical', fontsize=8)
plt.title('Task-1: Reward vs. Episode plots', fontsize=12)

# plt.grid()
plt.show()
 



# print (episode_rewards_all[29,:,-1])
for sensory_value in range(sensory_versions):	
	final_displacement[:,sensory_value] = episode_displacement_all[:,sensory_value,-1]
	# for mc_cntr in range(total_MC_runs):
		# pass_displacement_point[mc_cntr, sensory_value] = np.min(np.where(episode_displacement_all[mc_cntr, sensory_value,:]>=displacement_point))



# boxplot figure
s0 = final_displacement[:,0]
s1 = final_displacement[:,1]
s2 = final_displacement[:,2]
s = [s0,s1,s2]
plt.boxplot(s, whis='range', showfliers=True, showmeans=True, meanline=True, notch=True, patch_artist=True)
plt.xticks([1,2,3], sensory_info, fontsize=8)
plt.title('Final Reward', fontsize=8)
# plt.grid()
plt.show()
# print ('s0 std:', s0.std(), 's1 std:', s1.std(), 's2 std:', s2.std())


 # CLEAN BOXPLOT

plt.figure(figsize=(7,5))

Ns0= np.sort(s0, axis=0)
Ns1= np.sort(s1, axis=0)
Ns2= np.sort(s2, axis=0)
index = [0, 1, 2, 3,4, total_MC_runs-1, total_MC_runs-2, total_MC_runs-3, total_MC_runs-4]
index1 = [ total_MC_runs-1, total_MC_runs-2, total_MC_runs-3, total_MC_runs-4]
index2 = [0, 1, 2, 3,4,5,6,7,8]
clean_s0 =np.delete(Ns0, index)
clean_s1 =np.delete(Ns1, index)
# print(Ns1)
# print(clean_s1)
clean_s2 =np.delete(Ns2, index2)
clean_s = [clean_s0, clean_s1, clean_s2]



#Clean BOXPLOT

meanlineprops = dict(linestyle='--', linewidth=2, color='green')
medianlineprops =dict(linestyle='-', linewidth=2, color='red')
plt.boxplot(clean_s, whis='range', showfliers=True, showmeans=True, meanline= False, notch=True, patch_artist=True, medianprops = medianlineprops, meanprops=meanlineprops, widths = 0.25)
plt.plot([1,2,3],[clean_s0.mean(), clean_s1.mean(), clean_s2.mean()], c="green",linestyle='--', lw=1)

plt.xticks([1,2,3], sensory_info, fontsize=10)
plt.ylabel('Final Reward', fontsize=10)
plt.title('Task1: Final Reward', fontsize=10)
# plt.legend()
# plt.show()

colors = ['red', 'green']
lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]
labels = ['median', 'mean']
plt.legend(lines, labels, fontsize=10)
plt.ylim(-10, 2000)

plt.show()


F01, P01= stats.f_oneway(clean_s0, clean_s1)
print('clean' ,F01,P01)
# print('ANOVA01' , ANOVA01)   #F_onewayResult(statistic=0.04039321719253735, pvalue=0.8411302533815794)
# here were no statistically significant differences between group means as determined by one-way
F02, P02 = stats.f_oneway(clean_s0, clean_s2)
# print('ANOVA02', ANOVA02) #F_onewayResult(statistic=1.4583571345952087, pvalue=0.23009811284714493)
print('clean', F02,P02)

F12, P12 = stats.f_oneway(clean_s1, clean_s2)
# print('ANOVA12', ANOVA12) #F_onewayResult(statistic=2.0308635970310256, pvalue=0.15731008833875762)
print('clean', F12,P12)

'''

s0_mean = s0.mean(0)
s0_std = s0.std(0)
final_list0 = [x for x in s0 if (x > s0_mean - 2 * s0_std)]
# print (final_list0)
s1_mean = s1.mean(0)
s1_std =s1.std(0)
final_list1 = [x for x in s1 if (x > s1_mean - 2 * s1_std)]
# print (final_list1) 
s2_mean = s2.mean(0)
s2_std =s2.std(0)
final_list2 = [x for x in s2 if (x > s2_mean - 2 * s2_std)]
# print (final_list2)

final_list= [final_list0, final_list1, final_list2]
plt.boxplot(final_list)
plt.show()

'''


# one-way ANOVA
ANOVA01= stats.f_oneway(s0, s1)
print( 'ANOVA01', ANOVA01)   #F_onewayResult(statistic=0.8531299886534384, pvalue=0.3579370104660258)


ANOVA02 = stats.f_oneway(s0, s2)
print('ANOVA02', ANOVA02) #F_onewayResult(statistic=7.109697474837559, pvalue=0.008969409170012755)

ANOVA12 = stats.f_oneway(s1, s2)
print('ANOVA12', ANOVA12)  #F_onewayResult(statistic=3.4382729182718936, pvalue=0.06670950587490429)




# T-test
# Ttest10 = stats.ttest_ind(s0,s1, equal_var = False)

# print(Ttest10)

# Tukey HSD-teat
# mc = MultiComparison(s0, s1)
# result = mc.tukeyhsd()
 
# print(result)
# print(mc.groupsunique)


# average plot
x2=range(sensory_versions)
y2 = final_displacement.mean(0)
std2 = final_displacement.std(0)
# plt.plot(x2, y2,color='black',alpha=.1)

plt.plot(x2, y2, '--',color='black',alpha=.1)
# plt.fill_between(x2, y2-std2/2, y2+std2/2, alpha=0.25, edgecolor='C9', facecolor='C9')
plt.errorbar(x2, y2,yerr=std2/2,color='black',alpha=.2,animated=True)
for sensory_value in range(sensory_versions):
	plt.plot(x2[sensory_value], y2[sensory_value], 'o',alpha=.9, color=colorsys.hsv_to_rgb((8.75-sensory_value)/14,1,.75))
plt.xlabel('Sensory', fontsize=8)
plt.ylabel('Reward', fontsize=8)
plt.xticks(range(sensory_versions), sensory_info, fontsize=8)
plt.yticks( fontsize=8)
plt.title('Average Final Reward', fontsize=8)
# plt.grid()
plt.show()
