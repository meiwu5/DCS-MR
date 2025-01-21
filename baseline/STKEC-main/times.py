# Data for total time and average time
total_time = [60.482975999999994, 30.677854, 68.68728, 32.59152, 40.279199999999996, 45.344448, 44.91725100000001]
average_time = [1.3746478181818182, 2.359863923076923, 2.3685560344827588, 2.327995142857143, 2.237760944444444, 
                2.2672499000000004, 2.2458877999999998]

# Calculating sum of total time and mean of average time
sum_total_time = sum(total_time)
mean_average_time = sum(average_time) / len(average_time)

print(sum_total_time, mean_average_time)
