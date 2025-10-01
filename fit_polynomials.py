import numpy as np 
import matplotlib.pyplot as plt

x_min = -1 
x_max = 1 

# Generating a target polynomial. 
def f(x:float):
    ws = [0, 4, 2, -10]
    return sum([ws[i]*x**i for i in range(len(ws))])
xs = np.linspace(start=x_min, stop=x_max, num=1000)
ys = np.array([f(x) for x in xs])

# Sampling from the target. 
n = 10
X = np.array([-0.92, -0.73, -0.62, -0.2, 0.1, 0.05, 0.22, 0.55, 0.72, 0.92])
# X = np.random.uniform(low=x_min, high=x_max, size=n)
Y = np.array([f(x) + np.random.normal(0, 0.5) for x in X])


###############################
##### Fitting polynomials #####
###############################
degrees = 9

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

# Without regularization
ax[0].set_title("No regularization OLS")
ax[0].scatter(X, Y, label="samples")
for degrees in [9]:
    X_ols = np.array([[x**d for x in X] for d in range(degrees+1)]).T
    ols = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ Y
    preds = [sum([ols[d]*x**d for d in range(degrees+1)]) for x in xs]
    ax[0].plot(xs, preds, label=f"d={degrees}")
ax[0].plot(xs, ys, label="target")
ax[0].set_xlim(x_min, x_max)
ax[0].set_ylim(-4, 8)
ax[0].legend()

# With L2-regularization. 
ax[1].set_title("L2 Regularizazed OLS")
ax[1].scatter(X, Y, label="samples")
ax[1].plot(xs, ys, label="target")
for λ in [0.1]:
    X_ols = np.array([[x**d for x in X] for d in range(degrees+1)]).T
    reg_M = np.identity(degrees + 1)
    reg_M[0,0] = 0 # Leave the intercept unpenalized. 
    l2_ols = np.linalg.inv(X_ols.T @ X_ols + λ*reg_M) @ X_ols.T @ Y
    preds = [sum([l2_ols[d]*x**d for d in range(degrees+1)]) for x in xs]
    ax[1].plot(xs, preds, label=f"d={degrees}, λ={λ}")
ax[1].set_xlim(x_min, x_max)
ax[1].set_ylim(-4, 8)
ax[1].legend()
plt.savefig("l2_reg.png")
plt.show()

# With order-dependent regularization. 
degrees = 100
plt.title("Order dependent regularization.")
plt.plot(xs, ys, label="target")
plt.scatter(X, Y, label="samples")
for r in [1.5, 5]:
    X_ols = np.array([[x**d for x in X] for d in range(degrees+1)]).T
    # Basically, we do l2 regularization with strength increasing in the polynomial order.  
    I = np.identity(degrees + 1)
    I[0,0] = 0 # Leave the intercept unpenalized. 
    lambdas = np.array([0.001 * r**i for i in range(degrees + 1)])
    Lambda = lambdas * I
    beta = np.linalg.solve(X_ols.T @ X_ols + Lambda, X_ols.T @ Y)
    preds = [sum([beta[d]*x**d for d in range(degrees+1)]) for x in xs]
    plt.plot(xs, preds, label=f"d={degrees}, r={r}")
plt.xlim(left=x_min, right=x_max)
plt.ylim(bottom=-4, top=8)
plt.legend()
plt.savefig("order_dependent_reg.png",)