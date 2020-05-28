#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

import dynamic as dyn
import utils as ut

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def get_trim(aircraft, h, Ma, sm, km):
    """Calcul du trim pour un point de vol"""
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, h)
    Xe, Ue = dyn.trim(aircraft, {'va': va, 'h': h, 'gamma': 0})
    return Xe, Ue


def get_all_trims(aircraft, hs, Mas, sms, kms):
    """Calcul de trims pour une série de points de vol"""
    trims = np.zeros((len(hs), len(Mas), len(sms), len(kms), 3))
    for i, h in enumerate(hs):
        for j, Ma in enumerate(Mas):
            for k, sm in enumerate(sms):
                for l, km in enumerate(kms):
                    Xe, Ue = get_trim(aircraft, h, Ma, sm, km)
                    trims[i, j, k, l] = (Xe[dyn.s_a], Ue[dyn.i_dm], Ue[dyn.i_dth])
    return trims


def plot_all_trims(aircraft, hs, Mas, sms, kms, trims, filename=None):
    """Affichage des trims, tentative naïve"""
    margins = (0.03, 0.05, 0.98, 0.95, 0.2, 0.38)
    fig = ut.prepare_fig(window_title='Trims {}'.format(aircraft.name), figsize=(20.48, 10.24), margins=margins)

    m = 0
    for k, sm in enumerate(sms):
        for l, km in enumerate(kms):
            for i, h in enumerate(hs):
                for j, Ma in enumerate(Mas):
                    alpha, dphr, dth = trims[i, j, k, l]
                    fmt = 'alt {:5.0f} Ma {:.1f} sm {:.1f} km {:.1f} -> alpha {:5.2f} deg phr {:-5.1f} deg throttle {:.1f} %'
                    print(fmt.format(h, Ma, sm, km, ut.deg_of_rad(alpha), ut.deg_of_rad(dphr), 100 * dth))
            ax = plt.subplot(4, 3, 3 * m + 1)
            plt.plot(hs, ut.deg_of_rad(trims[:, 0, k, l, 0]))
            plt.plot(hs, ut.deg_of_rad(trims[:, 1, k, l, 0]))
            ut.decorate(ax, r'$sm {} \quad km {}$'.format(sm, km), r'altitude', r'$\alpha$ deg',
                        legend=['Mach {}'.format(Ma) for Ma in Mas])
            ax = plt.subplot(4, 3, 3 * m + 2)
            plt.plot(hs, ut.deg_of_rad(trims[:, 0, k, l, 1]))
            plt.plot(hs, ut.deg_of_rad(trims[:, 1, k, l, 1]))
            ut.decorate(ax, r'$sm {} \quad km {}$'.format(sm, km), r'altitude', r'$\delta_{phr}$ deg',
                        legend=['Mach {}'.format(Ma) for Ma in Mas])
            ax = plt.subplot(4, 3, 3 * m + 3)
            plt.plot(hs, trims[:, 0, k, l, 2] * 100)
            plt.plot(hs, trims[:, 1, k, l, 2] * 100)
            ut.decorate(ax, r'$sm {} \quad km {}$'.format(sm, km), r'altitude', 'throttle %',
                        legend=['Mach {}'.format(Ma) for Ma in Mas])
            m = m + 1
    if filename is not None:
        plt.savefig(filename, dpi=160)
    return fig


def plot_traj_trim(aircraft, h, Ma, sm, km):
    """Affichage d'une trajectoire avec un point de trim comme condition initiale"""
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, h)
    Xe, Ue = dyn.trim(aircraft, {'va': va, 'h': h, 'gamma': 0})
    time = np.arange(0., 100, 0.5)
    X = scipy.integrate.odeint(dyn.dyn, Xe, time, args=(Ue, aircraft))  # integration numérique de dyn.dyn
    dyn.plot(time, X)


def get_CL_Fmax_trim(aircraft, h, Ma):
    p, rho, T = ut.isa(h)
    va = dyn.va_of_mach(Ma, h)
    pdyn = 0.5 * rho * va ** 2
    return aircraft.m * aircraft.g / (pdyn * aircraft.S), dyn.propulsion_model([0, h, va, 0, 0, 0], [0, 1, 0, 0],
                                                                               aircraft)


def get_linearized_model(aircraft, h, Ma, sm, km):
    """Calcul numérique du modèle tangent linéarisée pour un point de trim"""
    aircraft.set_mass_and_static_margin(km, sm)
    va = dyn.va_of_mach(Ma, h)
    Xe, Ue = dyn.trim(aircraft, {'va': va, 'h': h, 'gamma': 0})
    A, B = ut.num_jacobian(Xe, Ue, aircraft, dyn.dyn)  # calcul du linéarisé tangent, A est une matrice 6x6
    poles, vect_p = np.linalg.eig(A[dyn.s_va:, dyn.s_va:])  # il ne ressort que 4 pôles
    return A, B, poles, vect_p


def plot_poles(aircraft, hs, Mas, sms, kms, filename=None):
    margins = (0.03, 0.05, 0.98, 0.95, 0.2, 0.38)
    fig = ut.prepare_fig(window_title='Poles {} (km:{})'.format(aircraft.name, kms), figsize=(20.48, 10.24),
                         margins=margins)
    plt.grid(True, color='k', linestyle='-', which='both', axis='both', linewidth=0.2)
    for i, h in enumerate(hs[::-1]):
        for j, Ma in enumerate(Mas):
            ax = plt.subplot(len(hs), len(Mas), i * len(Mas) + j + 1)
            for k, sm in enumerate(sms):
                for l, km in enumerate(kms):
                    A, B, poles, vect_p = get_linearized_model(aircraft, h, Ma, sm, km)
                    print('{}'.format(poles))
                    plt.plot(poles.real, poles.imag, '*', markersize=20, alpha=1.)
                    ut.decorate(ax, r'$h:{}m \quad  Ma:{} Km:{}$'.format(h, Ma, km),
                                legend=['ms: {}'.format(sm) for sm in sms])
    if filename is None:
        plt.savefig(filename, dpi=160)
    return fig


def plot_trims(aircraft, sms, kms, filename=None):
    machs = np.linspace(0.5, 0.8, 10)
    hs = [3000, 10000]

    def thrust(mach, h, ms, km):
        aircraft.set_mass_and_static_margin(km, ms)
        va = dyn.va_of_mach(mach, h)
        X, U = dyn.trim(aircraft, {'va': va, 'h': h, 'gamma': 0})
        return dyn.propulsion_model(X, U, aircraft)

    figure = ut.prepare_fig(None, 'Poussée en fonction du mach {name}'.format(name=aircraft.name),
                            margins=(0.05, 0.08, 0.98, 0.93, 0.2, 0.31))

    compteur = 1
    for i, km in enumerate(kms):
        for j, ms in enumerate(sms):
            ax = plt.subplot(2, 2, compteur)
            compteur += 1
            for h in hs:
                thrusts = [thrust(mach, h, ms, km) for mach in machs]
                plt.plot(machs, thrusts)
                ut.decorate(ax, 'Poussée (mach), ms : {}, km : {}'.format(ms, km), 'Mach', 'Thrust',
                            legend=['h : {}'.format(h) for h in hs])

    if filename is not None:
        plt.savefig(filename, dpi=160)
    return figure


if __name__ == "__main__":
    aircraft = dyn.Param_A321()
    hs, Mas = np.linspace(3000, 11000, 2), [0.4, 0.8]
    sms, kms = [0.2, 0.95], [0.1, 0.95]

    trims = get_all_trims(aircraft, hs, Mas, sms, kms)

    plot_all_trims(aircraft, hs, Mas, sms, kms, trims, 'plots/seance_2/{}_trim.png'.format(aircraft.get_name()))

    plot_trims(aircraft, sms, kms, filename='plots/seance_2/{} poussee - mach.png'.format(aircraft.get_name()))

    # plot_traj_trim(aircraft, 5000, 0.5, 0.2, 0.5)

    # plot_poles(aircraft, hs, Mas, sms, [kms[0]], 'seance_2/plots/{}_poles_1.png'.format(aircraft.get_name()))
    # plot_poles(aircraft, hs, Mas, sms, [kms[1]], 'seance_2/plots/{}_poles_2.png'.format(aircraft.get_name()))
    # plt.show()
