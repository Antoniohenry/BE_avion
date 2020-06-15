import matplotlib.pyplot as plt
import numpy as np

import dynamic
import utils


def second_regime(aircraft, figure=None):
    hs = np.linspace(3000, 11000, 5)
    fig = utils.prepare_fig(window_title='second regime')
    mas = np.linspace(0.3, 0.85, 100)
    U_max = [0, 1., 0, 0]
    colors = ['blue', 'red', 'black', 'pink', 'green']
    compteur = 0
    gamma = 0
    for h in hs:
        Xs = [dynamic.trim(aircraft, {'va': dynamic.va_of_mach(ma, h), 'gamma': gamma, 'h': h})[0] for ma in mas]
        Us = [dynamic.trim(aircraft, {'va': dynamic.va_of_mach(ma, h), 'gamma': gamma, 'h': h})[1] for ma in mas]
        pousse_necessaire = [dynamic.get_aero_forces_and_moments(X, U, aircraft)[1] for X, U in zip(Xs, Us)]
        pousse_max = [2 * dynamic.propulsion_model(X, U_max, aircraft) for X in Xs]
        plt.plot(mas, pousse_necessaire, '-', color=colors[compteur], label='h : {}m'.format(h))
        plt.plot(mas, pousse_max, '--', color=colors[compteur])
        compteur += 1
    plt.legend()
    plt.show()


if __name__ == '__main__':
    aircraft = dynamic.Param_A321()
    aircraft.set_mass_and_static_margin(0.95, 0.95)
    second_regime(aircraft, 'second regime')
