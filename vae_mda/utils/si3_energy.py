"""SI3 iceistate.F90 extensive ice energy for the new-ice restart case."""


def new_ice_energy_density(temperature_k, salinity_g_kg):
    """Return J/m3 of ice for SI3's constant-salinity two-layer regime.

    The deployed nn_icesal=2 profile is constant when bulk salinity >=4.5.
    The caller supplies the actual bulk salinity implied by the restart sv_i/v_i.
    Constants match NEMO phycst.F90, iceistate.F90 and eosbn2.F90.
    """
    melt_k = 273.15 - 0.054 * salinity_g_kg
    return 917.0 * (
        2067.0 * (melt_k - temperature_k)
        + 334000.0 * (1.0 - (melt_k - 273.15)
                      / min(temperature_k - 273.15, -1.e-20))
        - 3991.86795711963 * (melt_k - 273.15)
    )
