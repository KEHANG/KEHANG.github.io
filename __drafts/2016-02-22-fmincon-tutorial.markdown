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

#### Set up constraints

1. Mass balance

2. Kinetics

3. Variable bonds

[matlab-fmincon-tutorial]: http://www.mathworks.com/help/optim/ug/fmincon.html?refresh=true#busohxx-2

[recitation-problem]: https://stellar.mit.edu/S/course/10/sp16/10.37/courseMaterial/topics/topic3/lectureNotes/Recitation_Slides_Feb_22_2016/Recitation_Slides_Feb_22_2016.pdf
