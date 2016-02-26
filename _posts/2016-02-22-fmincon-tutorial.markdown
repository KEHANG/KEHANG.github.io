---
layout: post
title:  "Reactor Optimization with fmincon!"
date:   2016-02-22 14:00:51 -0500
categories: 10.37_TA
---
MIT 10.37 students and other chemical engineers in reactor enegineering might find this post interesting. Here I'm going to talk about how to apply `fmincon` to typical reactor engineering optimization problems.

### Basic `fmincon` usage

Consider the optimization problem below,

$$ \underset{X}{\text{minimize}}\ 100  \left(  x_2-x_1^2 \right) ^2 + \left( 1-x_1\right) ^2 $$

$$ s.t.\ x_1 + 2x_2 \leq 4 $$

$$  2x_1 + x_2 = 1 $$

$$  0 \leq x_1 \leq 0.4 $$

$$  0 \leq x_2 \leq 0.4 $$

it only takes three steps to translate into matlab code as shown below.

{% highlight matlab %}
% step1: define objective function
fun = @(x)100*(x(2)-x(1)^2)^2 + (1-x(1))^2; 

% step2: set up constraints
A = [1,2];
b = 4;
Aeq = [2, 1];
beq = 1;
lb = [0,0];
ub = [0.4, 0.4];

% step3: provide initial guess and call fmincon
x0 = [1,1];
x = fmincon(fun, x0, A, b, Aeq, beq,lb, ub)
{% endhighlight %}

There's an alternative way of translating constraints, instead of using `A`, `b`, `Aeq`, `beq`, `lb`, and `ub`, we can use `c` and `ceq`.

{% highlight matlab %}
% step1: define objective function
fun = @(x)100*(x(2)-x(1)^2)^2 + (1-x(1))^2; 

% step2: set up constraints
function [c,ceq] = constraints(x)
c(1) = x(1) + 2x(2) - 4;
c(2) = x(1) - 0.4;
c(3) = x(2) - 0.4;
c(4) = -x(1);
c(5) = -x(2);
ceq = 2x(1) + x(2) - 1;
return

A = [];
b = [];
Aeq = [];
beq = [];
lb = [];
ub = [];

% step3: provide initial guess and call fmincon
x0 = [1,1];
x = fmincon(fun, x0, A, b, Aeq, beq,lb, ub, @constraints)
{% endhighlight %}

[Matlab tutorial][matlab-fmincon-tutorial] has well covered typical ways of using `fmincon` to perform optimization. Please check it out if you want to know more.

### Reactor optimization example

Now with the basic knowledge of using `fmincon`, let's apply it to our reactor design domain. A good example is 2016-02-22's 10.37 [recitation problem][recitation-problem]. It has a typical reaction pathway where reactant goes to desired product, which however will react to form an undesired product. As chemical engineers, we should manipulate knobs, e.g., temperature, residence time, etc., in order to maximize molar yiled of desired product.

#### Define objective function 


As we discussed, here we want to maximize molar yield of `D` (diglyceride) by tuning $$T$$ (temperature), $$C_{T,0}$$, $$C_{CH_3O,0}$$, $$C_{CH_3OH,0}$$ and $$\tau$$. Putting it into mathematic term would be 

$$ \underset{T,\ C_{T,0},\ C_{CH_3O,0},\ C_{CH_3OH,0},\ \tau }{\text{minimize}}\ -  \frac{C_D}{C_{T,0}} $$

Translating it to matlab, we can define in below:

{% highlight matlab %}
function obj = objFun(x)
% x(1) = tau
% x(2) = temperature
% x(3) = [T]_0
% x(4) = [CH3O]_0
% x(5) = [CH3OH]_0
% x(6) = [T]
% x(7) = [CH3O]
% x(8) = [CH3OH]
% x(9) = [D]
% x(10) = k (the kinetic coefficient)

obj = -x(9)/x(3); % molar yield of D
end
{% endhighlight %}

#### Set up constraints

Almost all the real-life chemical engineering optimization problems have certain number of constraints. They are usually from `mass balance`, `kinetics rules`, `physical bounds`, etc.

- Mass balance

For convenience, we assume volumetric flowrate is not changed from inlet to outlet. so we have the following equations for major species' mass balance in CSTR.

$$  0 = - C_D + (3C_T - 2C_D)*kC_{CH_3O}\tau $$

$$  0 = - C_{T,0} - C_T - 3C_T*kC_{CH_3O}\tau $$

$$  0 = C_{CH_3O,0} - C_{CH_3O} $$

$$  0 = C_{CH_3OH,0} - C_{CH_3OH} - (3C_T - 2C_D)*kC_{CH_3O}\tau $$

- Kinetics rules

Kinetics coefficient `k` has temperature dependence according to Arrhenius law.

$$  0 = k - A*exp(\frac{-E_a}{RT}) $$

- Physical bounds

Physical bounds are those constraints that seem obvious to you but can largely reduce the search space for Matlab. So in order to get correct answer more quickly, we should provide as much extra information of physics as possible.

For instance, the concentrations and residence time can not be negative, and temperature should always be grater than 273 K (otherwise frozen). Here we will add another constraint where temperature cannot go beyound 350 C because of heating power limitation.

In summary we have,

$$  0 \leq C_T $$

$$  0 \leq C_{CH3O} $$

$$  0 \leq C_{CH3OH} $$

$$  0 \leq \tau $$

$$  273 K \leq T \leq 623 K $$

Put all these constraints together in Matlab using `c` and `ceq`,

{% highlight matlab %}
function [c, ceq] = allcon(x)
% params
% assign some easy numbers for
% Arrhenius expression params
A = exp(2);
R = 8.314;
Ea = R*600;

% ineqn
c(1) = -x(1);
c(2) = -x(2) + 273;
c(3) = -x(3);
c(4) = -x(4);
c(5) = -x(5);
c(6) = -x(6);
c(7) = -x(7);
c(8) = -x(8);
c(9) = -x(9);
c(10) = x(2) - 623;

% eqn
ceq(1) = x(10) - A*exp(-Ea/R/x(2));
ceq(2) = -x(9) + (3*x(6) - 2*x(9))*x(10)*x(7)*x(1);
ceq(3) = x(3) - x(6) - 3*x(6)*x(10)*x(7)*x(1);
ceq(4) = x(4) - x(7);
ceq(5) = x(5) - x(8) - 3*x(6)*x(10)*x(7)*x(1) - 2*x(9)*x(10)*x(7)*x(1);
end
{% endhighlight %}

#### call `fmincon` to optimize

Simply run the optimization by calling `fmincon` and feeding `objFun` and `allcon` we defined early.

{% highlight matlab %}
% give the following initial guess for the optimal solution
% tau = 1
% temperature = 300K
% [T]0 = 4
% [C3H0]0 = 1
% [CH3OH]0 = 6
% [T] = 1
% [CH3O] = 1
% [CH3OH] = 1
% [D] = 1
% k = 1
x0 = [1,300,4,1,6,1,1,1,1,1];
[x,fval] = fmincon(@objFun, x0, [], [], [], [], [], [], @allcon)
{% endhighlight %}

Eventually you will have optimal yield of diglyceride to be 30%. You can [find the Matlab codes here]({{ site.github.url }}/assets/fmincon_post/fmincon_reactor_optimization.zip).

[matlab-fmincon-tutorial]: http://www.mathworks.com/help/optim/ug/fmincon.html?refresh=true#busohxx-2

[recitation-problem]: https://stellar.mit.edu/S/course/10/sp16/10.37/courseMaterial/topics/topic3/lectureNotes/Recitation_Slides_Feb_22_2016/Recitation_Slides_Feb_22_2016.pdf
