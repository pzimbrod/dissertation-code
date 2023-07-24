from dolfinx.fem import Function
from ufl import FacetNormal, TrialFunction, dot, jump,avg, Form

def upwind(velocity: Function,n: FacetNormal, trial: TrialFunction) -> Form:
    vel_n = 0.5*(dot(velocity, n) + abs(dot(velocity, n)))
    return jump(vel_n * trial)

def lax_friedrichs(velocity: Function,n: FacetNormal, trial: TrialFunction) -> Form:
    v_max = max(max(velocity),0)
    return dot(avg(velocity * trial),n('+')) + 0.5 * v_max * jump(trial)

def HLLE(velocity: Function,n: FacetNormal, trial: TrialFunction) -> Form:
    v_max = max(max(velocity),0)
    v_min = min(min(velocity),0)
    return v_max/(v_max-v_min) * dot(velocity('+'),n('+'))*trial('+') - \
          v_min * dot(velocity('-'),n('+'))*trial('-') - v_max * v_min * jump(trial)