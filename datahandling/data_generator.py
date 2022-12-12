import argparse
# import os

# c4-c2, c2-c2, c4-quick, c2-quick AMDc-AMDs
#
# c4-c2, c4-quick, c2-quick, AMDc-AMDc


def get_args():
    parser = argparse.ArgumentParser(
        description='DNS/LES data file generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-ijk',
                        '--mesh',
                        nargs='+',
                        default=[192, 158],
                        help='Number of mesh points in the ijk directions')

    parser.add_argument(
        '-mesh-disc',
        type=str,
        help="Number of points in each direction"
    )
    parser.add_argument(
        '-ij-size',
        type=float,
        nargs="+",
        default=[0.0375055897, 0.0125018632]
    )

    parser.add_argument(
        '-kcn',
        '--k-coord-name',
        type=str,
        default="coord_k_",
        help="K direction coordinates names"
    )
    parser.add_argument(
        '-s',
        '--splitting',
        nargs='+',
        type=int,
        default=[16, 8, 1],
        help="Post splitting"
    )
    parser.add_argument(
        '--ts',
        '--timestep',
        type=int,
        dest="timestep",
        default=1,
        help="Timestep"
    )

    parser.add_argument(
        '-np',
        '--nb-steps-max',
        type=int,
        default=9999999
    )

    parser.add_argument(
        '-nck',
        '--n-coarsen-k',
        type=int,
        default=4,
        help="Number of coarsening operations in the k direction"
    )
    parser.add_argument(
        '-ncu', '--n-coarsen-uniform',
        type=int,
        help="Number of uniform coarsening operations"
    )
    parser.add_argument(
        '-cu', '--coarsen-uniform',
        type=str,
        nargs="+",
        help="Uniform coarsen operators"
    )
    parser.add_argument(
        '-ci', '--coarsen-uniform-i',
        type=int,
        nargs="+",
        help="Uniform coarsening in i direction"
    )

    parser.add_argument(
        '-cj', '--coarsen-uniform-j',
        type=int,
        nargs="+",
        help="Uniform coarsening in j direction"
    )
    parser.add_argument(
        '-ck', '--coarsen-uniform-k',
        type=int,
        nargs="+",
        help="Uniform coarsening in k direction"
    )

    parser.add_argument(
        '-gs', '--ghost-size',
        type=int,
        default=2,
        help="Ghost size for multigrid"
    )
    parser.add_argument(
        '-psp', '--pre-smooth-steps',
        type=int,
        nargs="+",
        default=None,
        # [5, 5, 5, 5, 8, 8, 8],
        help='Number of pre-smoothing sweeps'
    )

    parser.add_argument(
        '-ss', '--smooth-steps',
        type=int,
        nargs="+",
        default=None,
        # [5, 5, 5, 5, 8, 8, 8],
        help='Number of post-smoothing steps'
    )
    parser.add_argument(
        '-rj', '--relax-jacobi',
        type=float,
        nargs="+",
        default=None,
        # [.69, .69, .69, .69, .65, .65, .65],
        help='Jacobi relaxation coefficient'
    )

    parser.add_argument(
        '-cs', '--coarse-solver',
        type=float,
        default=.5e-6,
        help='Coarse solver maximum error'
    )
    parser.add_argument(
        '-cr', '--check-residu',
        type=float,
        default=0
    )
    parser.add_argument(
        '-mgth', '--multigrid-threshold',
        type=float,
        default=2.5e-6,
        help="Multigrid precision threshold"
    )
    parser.add_argument(
        '-nfmg', '--nb-full-mg-steps',
        type=int,
        nargs="+",
        default=[2, 4, 1],
        help="Number of full multigrid steps"
    )
    parser.add_argument(
        '-prec', '--precond',
        type=str,
        default="mixed",
        help="Coarse solver preconditionner"
    )
    parser.add_argument(
        '-sp', '--solver-precision',
        type=str,
        default="mixed",
        help="Type of precision for solver"
    )

    parser.add_argument(
        '-ts-facsec', '--timestep-facsec',
        type=float,
        default=1.,
        help="Will fill this later"
    )
    parser.add_argument(
        '-tinit',
        type=float,
        default=0.0,
        help="Initial time"
    )

    parser.add_argument(
        '-dt-post', '--dt-post',
        type=int,
        default=2000,
        help="Number of time steps to save a statistics/spacial mean file"
    )
    parser.add_argument(
        '-dtsauv', '--dt-sauv',
        type=int,
        default=10000,
        help="Number of time steps to save snapshots of the flow"
    )

    parser.add_argument(
        '-cv-rho', '--convection-rho',
        type=str,
        default="",
        help="Type of convection for rho"
    )

    parser.add_argument(
        '-turb-visc', '--turbulent-viscosity',
        type=str,
        default="",
        nargs="+",
        help="Type(s) of turbulent viscosity models"
    )

    parser.add_argument(
        '-turb-diff', '--turbulent-diffusivity',
        type=str,
        default="",
        nargs="+",
        help="Type(s) of turbulent diffusivity models"
    )

    parser.add_argument(
        '-les-form', '--les-formulation',
        type=str,
        default="Favre",
        help="Large Eddy Simulation formulation"
    )

    parser.add_argument(
        '-t-kmin', "--imposed-t-kmin",
        type=float,
        default=900.0,
        help="Imposed temperature at bottom wall"
    )

    parser.add_argument(
        '-t-kmax', '--imposed-t-kmax',
        type=float,
        default=1300.0,
        help="Imposed temperature at top wall"
    )
    parser.add_argument(
        '-p-th', '--p-th-init',
        type=float,
        default=10e5,
        help=r"Imposed initial P_{th}"
    )
    parser.add_argument(
        '-cp', '--cp',
        type=float,
        default=1155.0,
        help="Heat capacity of fluid"
    )

    parser.add_argument(
        '-gamma', '--gamma',
        type=float,
        default=1.4
    )

    parser.add_argument(
        '-acs', '--acceleration-source-term',
        type=float,
        default=15349.8795619994,
        help="Acceleration source term in momentum equation "
        "to impose no-slip condition"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mesh_disc:
        if args.mesh_disc == "AAA":
            ni, nj = 256, 192
            n_coarsening_operations = 4 + 3
            npi, npj, npk = 16, 8, 1
            ci, cj, ck = (1, 2, 2), (2, 2, 2), (2, 2, 2)
            n_coarsen_k = 4
            n_coarsen_uniform = 3
        if args.mesh_disc == "BAB":
            ni, nj = 192, 128
            n_coarsening_operations = 4 + 3
            npi, npj, npk = 16, 8, 1
            ci, cj, ck = (1, 2, 2), (1, 2, 2), (2, 2, 2)
            n_coarsen_k = 4
            n_coarsen_uniform = 3
        if args.mesh_disc == "CAC":
            ni, nj = 128, 72
            n_coarsening_operations = 4 + 2
            npi, npj, npk = 8, 2, 1
            ci, cj, ck = (2, 2), (2, 2),  (2, 2)
            n_coarsen_k = 4
            n_coarsen_uniform = 2
        # n_coarsening_operations = args.n_coarsen_k + args.n_coarsen_uniform
        # else:
        #     ni, nj = args.ijk
    dns_contents = "IJK_Grid_Geometry grid_geom2\n{\n nbelem_i " + str(
        ni) + "\n"
    sizei, sizej = args.ij_size
    k_coord_name = args.k_coord_name
    # npi, npj, npk = args.splitting  # Splitting
    # ci = args.ci
    # cj = args.cj
    # ck = args.ck
    dns_contents += f" nbelem_j {nj}\n"
    dns_contents += f" uniform_domain_size_i {sizei}\n"
    dns_contents += f" uniform_domain_size_j {sizej}\n"
    dns_contents += f" file_coord_k {k_coord_name}0.txt\n"
    dns_contents += " perio_i\n perio_j\n}\n\n"

    dns_contents += "IJK_Splitting grid_splitting2\nLire grid_splitting2\n{"
    dns_contents += " ijk_grid_geometry geid_geom2\n"
    dns_contents += f" nproc_i {npi}\n nproc_j {npj}\n nprock {npk}\n"
    dns_contents += "}"

    dns_contents += """\n
IJK_Splitting post_splitting
Lire post_splitting
{
  ijk_grid_geometry grid_geom2
  nproc_i 1
  nproc_j 1
  nproc_k 1
}\n
    """

    dns_contents += "dns_qc_double\n{\n ijk_splitting grid_splitting2\n"
    dns_contents += f" timestep {args.timestep}\n"
    dns_contents += f"nb_pas_dt_max {args.nb_steps_max}\n"
    dns_contents += " multigrid_solver {"
    # n_coarsening_operations = args.n_coarsen_k + args.n_coarsen_uniform
    dns_contents += f"  coarsen_operators {n_coarsening_operations}" + "}\n"
    for i in range(1, args.n_coarsen_k+1):
        dns_contents += "  Coarsen_Operator_K { file_z_coord " + \
            f"{k_coord_name}{i}.txt" + " }\n"
    #  This will come in later iterations
    # if len(args.ncu) > len(args.ci) or len(args.ncu) > len(args.cj) or len(args.ncu) > len(args.ck):

    for i in range(n_coarsen_uniform):
        if ci[i] == cj[i] and cj[i] == ck[i]:
            dns_contents += "  Coarsen_Operator_Uniform { }\n"
        else:
            dns_contents += "  Coarsen_Operator_Uniform {" + \
                f"coarsen_i {ci[i]} coarsen_j {cj[i]} coarsen_k {ck[i]}" + "}\n"
    dns_contents += "\n"

    if not args.pre_smooth_steps or not args.smooth_steps or args.rj:
        if n_coarsening_operations < 5:
            args.pre_smooth_steps = [5 for _ in range(n_coarsening_operations)]
            args.smooth_steps = args.pre_smooth_steps
            args.rj = [.69 for _ in range(n_coarsening_operations)]
        else:
            args.pre_smooth_steps = [5 for _ in range(
                4)] + [8 for _ in range(n_coarsening_operations - 4)]
            args.smooth_steps = args.pre_smooth_steps
            args.rj = [.69 for _ in range(
                4)] + [.65 for _ in range(n_coarsening_operations-4)]
    dns_contents += f"  ghost_size {args.ghost_size}\n"
    pre_smooth_steps = " ".join(str(x) for x in args.pre_smooth_steps)
    smooth_steps = " ".join(str(x) for x in args.smooth_steps)
    dns_contents += f"  pre_smooth_steps {n_coarsening_operations} {pre_smooth_steps}\n"
    dns_contents += f"  smooth_steps {n_coarsening_operations} {smooth_steps}\n"

    relax_jacobi = " ".join(str(x) for x in args.rj)
    dns_contents += f"  relax_jacobi {n_coarsening_operations} {relax_jacobi}\n"
    dns_contents += "  solveur_grossier GCP { seuil " + \
        str(args.coarse_solver) + " precond ssor { omega 1.5 } }\n"
    dns_contents += f"  check_residu {args.check_residu}\n"
    dns_contents += f"  seuil {args.multigrid_threshold}\n"
    full_mg_steps = " ".join(str(x) for x in args.nb_full_mg_steps)
    dns_contents += f"  nb_full_mg_steps {full_mg_steps}\n"
    dns_contents += f"  solver_precondition {args.precond}\n"

    dns_contents += " }\n"

    timestep_facsec = args.timestep_facsec

    dns_contents += f" timestep_facsec {timestep_facsec}\n"
    dns_contents += f"\n tinit {args.tinit}"

    dns_contents += """

  expression_t_init   2.68867*10^15*z^5-4.01903*10^13*z^4+2.15984*10^11*z^3-5.00048*10^8*z^2+490671*z+900
  expression_vx_init  95*(1-((((z-0.0029846)/0.0029846)^2)^0.5)^9)
  expression_vy_init  0.
  expression_vz_init  (sin((x-0.0375055897)/0.0375055897*6.28318530717959)*sin(y/0.0125018632*6.28318530717959)+sin((x-0.0375055897)/0.0375055897*6.28318530717959*6)*sin(y/0.0125018632*6.28318530717959*4)+sin((x-0.0375055897)/0.0375055897*6.28318530717959*2)*sin(y/0.0125018632*6.28318530717959*8))*z*(0.0059692-z)*4/(0.0059692*0.0059692)
  \n
    """

    dns_contents += f" dt_post {args.dt_post}\n"
    dns_contents += f" dt_sauvegarde {args.dt_sauv}\n"

    dns_contents += " check_stop_file stop_file\n"

    if args.convection_rho and not args.convection_rho.lower() == "quick":
        dns_contents += f" {args.convection_rho}"

    if args.turbulent_viscosity:
        dns_contents += " turbulent_viscosity\n"
        dns_contents += f" turbulent_viscosity_model {args.turbulent_viscosity[0]}\n"
        dns_contents += f" turbulent_viscosity_model_constant {args.turbulent_viscosity[1]}\n\n"

    if args.turbulent_diffusivity:
        dns_contents += " turbulent_diffusivity\n"
        dns_contents += f" turbulent_diffusivity_model {args.turbulent_diffusivity[0]}\n"
        dns_contents += f" turbulent_diffusivity_model_constant {args.turbulent_diffusivity[1]}\n\n\n"

    dns_contents += f"""

  t_paroi_impose_kmin {args.imposed_t_kmin}
  t_paroi_impose_kmax {args.imposed_t_kmax}
  p_thermo_init       {args.p_th_init}
  cp                  {args.cp}
  gamma               {args.gamma}

  terme_source_acceleration_constant      {args.acceleration_source_term}

    """
    dns_contents += "\n}"

    with open("dns.data", "w") as f:

        f.write(dns_contents)
