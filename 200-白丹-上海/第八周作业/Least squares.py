data=[[1,6],[2,5],[3,7],[4,10]]

sum_x=sum(point[0] for point in data)
sum_y=sum(point[1] for point in data)
sum_xy=sum(point[0]*point[1] for point in data)
sum_x2=sum(point[0]**2 for point in data)
n=len(data)

k=(n*sum_xy-sum_x*sum_y)/(n*sum_x2-sum_x*sum_x)
b=sum_y/n-(k*sum_x)/n

print(f"最小二乘法y={k}x+{b}")
