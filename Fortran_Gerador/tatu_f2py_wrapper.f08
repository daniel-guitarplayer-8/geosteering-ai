!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! tatu_f2py_wrapper.f08 — Interface f2py para o simulador PerfilaAnisoOmp
!
! Versão 9.0: suporte a F5, F7, F6 (compensação midpoint) e Filtro Adaptativo.
!
! Módulo wrapper que expõe o simulador EM 1D TIV a Python via f2py (NumPy).
! Retorna os arrays de saída diretamente (sem I/O de disco), permitindo
! integração eficiente com o pipeline DL geosteering_ai/ v2.0.
!
! Uso em Python (básico, sem F5/F7):
!   import tatu_f2py
!   zrho, cH = tatu_f2py.simulate(nf, freq, ntheta, theta, h1, tj,
!                                   nTR, dTR, p_med, n, resist, esp, nmmax)
!
! Uso com F5/F7:
!   zrho, cH, cH_tilted = tatu_f2py.simulate_v8(
!       nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, n, resist, esp, nmmax,
!       use_arb_freq, use_tilted, n_tilted, beta_tilt, phi_tilt)
!
! Onde:
!   nmmax = ceil(tj / (p_med * cos(theta_min * pi/180))) para ângulos < 90°
!   zrho(nTR, ntheta, nmmax, nf, 3) — resistividades aparentes
!   cH(nTR, ntheta, nmmax, nf, 9)   — tensor EM completo (complex128)
!   cH_tilted(nTR, ntheta, nmmax, nf, n_tilted) — respostas tilted (complex128, F7)
!
! Compatível com OpenMP: OMP_NUM_THREADS controlado via variável de ambiente.
!
! Ref: docs/reference/analise_evolucao_simulador_fortran_python.md §2
!      docs/reference/documentacao_simulador_fortran.md §6.5.2c
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
module tatu_wrapper
  use parameters
  use filterscommonbase
  use utils
  use magneticdipoles
  use omp_lib
  use DManisoTIV, only: compute_jacobian_fd   ! F10 — reutilizado em simulate_v10_jacobian
  implicit none
contains

subroutine simulate(nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, &
                    n, resist, esp, nmmax, zrho_out, cH_out)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Simula o campo EM 1D TIV para múltiplos pares T-R, ângulos e frequências.
  ! Retorna arrays NumPy diretamente (sem escrita em disco).
  !
  ! NOTA: Interface básica (v7.0 compatível). Usa SEMPRE o filtro Werthmuller
  !   (201 pontos). Para seleção de filtro (Kong/Anderson), F5/F7, ou F6,
  !   use simulate_v8() que aceita filter_type_in como parâmetro.
  !
  ! INPUT:
  !   nf          : número de frequências (integer)
  !   freq(nf)    : frequências em Hz (real64)
  !   ntheta      : número de ângulos de inclinação (integer)
  !   theta(ntheta): ângulos em graus (real64)
  !   h1          : altura do primeiro ponto-médio T-R (real64)
  !   tj          : tamanho da janela de investigação (real64)
  !   nTR         : número de pares T-R (integer)
  !   dTR(nTR)    : espaçamentos T-R em metros (real64)
  !   p_med       : passo entre medidas em metros (real64)
  !   n           : número de camadas geológicas (integer)
  !   resist(n,2) : resistividades (horizontal, vertical) por camada (real64)
  !   esp(n)      : espessuras das camadas (real64)
  !   nmmax       : número máximo de medidas por ângulo (integer, pré-calculado pelo caller)
  !
  ! OUTPUT:
  !   zrho_out(nTR, ntheta, nmmax, nf, 3) : resistividades aparentes (real64)
  !   cH_out(nTR, ntheta, nmmax, nf, 9)   : tensor EM completo (complex128)
  !
  ! NOTA: nmmax deve ser pré-calculado em Python como:
  !   nmmax = max(ceil(tj / (p_med * cos(theta[i] * pi / 180))) for i in range(ntheta))
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  implicit none
  ! Argumentos de entrada
  integer, intent(in) :: nf, ntheta, nTR, n, nmmax
  real(dp), intent(in) :: freq(nf), theta(ntheta), dTR(nTR), resist(n,2), esp(n)
  real(dp), intent(in) :: h1, tj, p_med
  ! Argumentos de saída
  !f2py intent(in) :: nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, n, resist, esp, nmmax
  !f2py intent(out) :: zrho_out, cH_out
  !f2py depend(nf) :: freq
  !f2py depend(ntheta) :: theta
  !f2py depend(nTR) :: dTR
  !f2py depend(n) :: resist, esp
  !f2py depend(nTR, ntheta, nmmax, nf) :: zrho_out, cH_out
  real(dp), intent(out) :: zrho_out(nTR, ntheta, nmmax, nf, 3)
  complex(dp), intent(out) :: cH_out(nTR, ntheta, nmmax, nf, 9)

  ! Variáveis locais (idênticas a perfila1DanisoOMP)
  integer :: i, j, k, itr
  integer, parameter :: npt = 201
  real(dp) :: thetamin, thetaplu, thetarad, del
  real(dp) :: z1, pz, posTR(6), ang, seno, coss, px, Lsen, Lcos
  real(dp) :: Tx_l, Ty_l, Tz_l, x_l, y_l, z_l, r_k, omega_i
  complex(dp) :: zeta_i
  character(5) :: dipolo
  integer, dimension(:), allocatable :: nmed
  real(dp), dimension(:), allocatable :: h, prof, krJ0J1_l, wJ0_l, wJ1_l
  real(dp), dimension(:,:), allocatable :: krwJ0J1, zrho, eta_shared
  complex(dp), dimension(:,:), allocatable :: cH_local
  real(dp), dimension(:,:,:), allocatable :: z_rho1
  complex(dp), dimension(:,:,:), allocatable :: c_H1
  integer :: maxthreads, num_threads_k, num_threads_j, ii, t, tid
  logical :: nested_enabled
  type(thread_workspace), allocatable :: ws_pool(:)
  ! Fase 4 — cache com dimensão ntheta (elimina race condition no outer parallel do k)
  complex(dp), allocatable :: u_cache(:,:,:,:), s_cache(:,:,:,:)
  complex(dp), allocatable :: uh_cache(:,:,:,:), sh_cache(:,:,:,:)
  complex(dp), allocatable :: RTEdw_cache(:,:,:,:), RTEup_cache(:,:,:,:)
  complex(dp), allocatable :: RTMdw_cache(:,:,:,:), RTMup_cache(:,:,:,:)
  complex(dp), allocatable :: AdmInt_cache(:,:,:,:)

  del = 1.d-6
  zrho_out = 0.d0
  cH_out = cmplx(0.d0, 0.d0, kind=dp)

  allocate(nmed(ntheta))
  z1 = -h1
  do i = 1, ntheta
    thetamin = theta(i) - del
    thetaplu = theta(i) + del
    if ((thetamin > 0.d0 .and. thetaplu < 9.d1) .or. (thetamin < 0.d0 .and. thetaplu > 0.d0)) then
      thetarad = theta(i) * pi / 18.d1
      pz = p_med * cos(thetarad)
      nmed(i) = ceiling(tj / pz)
    elseif (thetamin <= 9.d1 .and. thetaplu >= 9.d1 .and. dabs(tj) > del) then
      nmed(i) = ceiling(tj / p_med) + 1
    else
      nmed(i) = 1
    end if
  end do

  call J0J1Wer(npt, krJ0J1_l, wJ0_l, wJ1_l)
  allocate(krwJ0J1(npt,3))
  do i = 1, npt
    krwJ0J1(i,:) = (/ krJ0J1_l(i), wJ0_l(i), wJ1_l(i) /)
  end do

  call sanitize_hprof_well(n, esp, h, prof)
  dipolo = 'hmdxy'

  ! Configuração OpenMP (idêntica a perfila1DanisoOMP)
  call omp_set_max_active_levels(2)
  nested_enabled = (omp_get_max_active_levels() >= 2)
  maxthreads = omp_get_max_threads()
  num_threads_k = max(1, min(ntheta, maxthreads))
  num_threads_j = max(1, maxthreads / num_threads_k)

  ! Alocação do workspace pool (Fase 3 + 3b)
  allocate(ws_pool(0:maxthreads-1))
  do t = 0, maxthreads-1
    allocate(ws_pool(t)%Tudw(npt, n), ws_pool(t)%Txdw(npt, n))
    allocate(ws_pool(t)%Tuup(npt, n), ws_pool(t)%Txup(npt, n))
    allocate(ws_pool(t)%TEdwz(npt, n), ws_pool(t)%TEupz(npt, n))
    allocate(ws_pool(t)%Mxdw(npt), ws_pool(t)%Mxup(npt))
    allocate(ws_pool(t)%Eudw(npt), ws_pool(t)%Euup(npt))
    allocate(ws_pool(t)%FEdwz(npt), ws_pool(t)%FEupz(npt))
  end do

  ! Alocação dos caches Fase 4 — com dimensão ntheta (thread-safe)
  allocate(u_cache(npt, n, nf, ntheta), s_cache(npt, n, nf, ntheta))
  allocate(uh_cache(npt, n, nf, ntheta), sh_cache(npt, n, nf, ntheta))
  allocate(RTEdw_cache(npt, n, nf, ntheta), RTEup_cache(npt, n, nf, ntheta))
  allocate(RTMdw_cache(npt, n, nf, ntheta), RTMup_cache(npt, n, nf, ntheta))
  allocate(AdmInt_cache(npt, n, nf, ntheta))
  allocate(eta_shared(n, 2))
  do ii = 1, n
    eta_shared(ii, 1) = 1.d0 / resist(ii, 1)
    eta_shared(ii, 2) = 1.d0 / resist(ii, 2)
  end do

  allocate(zrho(nf, 3), cH_local(nf, 9))
  allocate(z_rho1(nmmax, nf, 3), c_H1(nmmax, nf, 9))
  zrho = 0.d0
  cH_local = cmplx(0.d0, 0.d0, kind=dp)
  z_rho1 = 0.d0
  c_H1 = cmplx(0.d0, 0.d0, kind=dp)

  ! Loop principal: multi-TR × theta × medidas
  do itr = 1, nTR
    !$omp parallel do schedule(dynamic) num_threads(num_threads_k) &
    !$omp&        if(ntheta > 1) &
    !$omp&        private(k,ang,seno,coss,px,pz,Lsen,Lcos,r_k,omega_i,zeta_i,ii) &
    !$omp&        firstprivate(z_rho1,c_H1)
    do k = 1, ntheta
      ang = theta(k) * pi / 18.d1
      seno = sin(ang)
      coss = cos(ang)
      px = p_med * seno
      pz = p_med * coss
      Lsen = dTR(itr) * seno
      Lcos = dTR(itr) * coss

      r_k = dTR(itr) * dabs(seno)
      do ii = 1, nf
        omega_i = 2.d0 * pi * freq(ii)
        zeta_i = cmplx(0.d0, 1.d0, kind=dp) * omega_i * mu
        call commonarraysMD(n, npt, r_k, krwJ0J1(:,1), zeta_i, h, eta_shared, &
                            u_cache(:,:,ii,k), s_cache(:,:,ii,k), &
                            uh_cache(:,:,ii,k), sh_cache(:,:,ii,k), &
                            RTEdw_cache(:,:,ii,k), RTEup_cache(:,:,ii,k), &
                            RTMdw_cache(:,:,ii,k), RTMup_cache(:,:,ii,k), &
                            AdmInt_cache(:,:,ii,k))
      end do

      !$omp parallel do schedule(guided, 16) &
      !$omp&        num_threads(merge(maxthreads, num_threads_j, ntheta == 1)) &
      !$omp&        default(shared) &
      !$omp&        private(j, x_l, y_l, z_l, Tx_l, Ty_l, Tz_l, posTR, tid) &
      !$omp&        firstprivate(zrho, cH_local)
      do j = 1, nmed(k)
        x_l  = 0.d0 + (j-1) * px - Lsen / 2
        y_l  = 0.d0
        z_l  = z1 + (j-1) * pz - Lcos / 2
        Tx_l = 0.d0 + (j-1) * px + Lsen / 2
        Ty_l = 0.d0
        Tz_l = z1 + (j-1) * pz + Lcos / 2
        posTR = (/Tx_l, Ty_l, Tz_l, x_l, y_l, z_l/)
        if (ntheta == 1) then
          tid = omp_get_thread_num()
        else
          tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
        end if
        call fieldsinfreqs_cached_ws(ws_pool(tid), ang, nf, freq, posTR, dipolo, npt, &
                                      krwJ0J1, n, h, prof, resist, eta_shared, &
                                      u_cache(:,:,:,k),  s_cache(:,:,:,k),    &
                                      uh_cache(:,:,:,k), sh_cache(:,:,:,k),   &
                                      RTEdw_cache(:,:,:,k), RTEup_cache(:,:,:,k), &
                                      RTMdw_cache(:,:,:,k), RTMup_cache(:,:,:,k), &
                                      AdmInt_cache(:,:,:,k),                  &
                                      zrho, cH_local)
        z_rho1(j,:,:) = zrho
        c_H1(j,:,:) = cH_local
      end do
      !$omp end parallel do
      zrho_out(itr, k, 1:nmed(k), :, :) = z_rho1(1:nmed(k), :, :)
      cH_out(itr, k, 1:nmed(k), :, :) = c_H1(1:nmed(k), :, :)
    end do
    !$omp end parallel do
  end do

  ! Desalocação
  deallocate(zrho, cH_local, z_rho1, c_H1)
  if (allocated(krwJ0J1)) deallocate(krwJ0J1)
  do t = 0, maxthreads-1
    if (allocated(ws_pool(t)%Tudw)) deallocate(ws_pool(t)%Tudw)
    if (allocated(ws_pool(t)%Txdw)) deallocate(ws_pool(t)%Txdw)
    if (allocated(ws_pool(t)%Tuup)) deallocate(ws_pool(t)%Tuup)
    if (allocated(ws_pool(t)%Txup)) deallocate(ws_pool(t)%Txup)
    if (allocated(ws_pool(t)%TEdwz)) deallocate(ws_pool(t)%TEdwz)
    if (allocated(ws_pool(t)%TEupz)) deallocate(ws_pool(t)%TEupz)
    if (allocated(ws_pool(t)%Mxdw)) deallocate(ws_pool(t)%Mxdw)
    if (allocated(ws_pool(t)%Mxup)) deallocate(ws_pool(t)%Mxup)
    if (allocated(ws_pool(t)%Eudw)) deallocate(ws_pool(t)%Eudw)
    if (allocated(ws_pool(t)%Euup)) deallocate(ws_pool(t)%Euup)
    if (allocated(ws_pool(t)%FEdwz)) deallocate(ws_pool(t)%FEdwz)
    if (allocated(ws_pool(t)%FEupz)) deallocate(ws_pool(t)%FEupz)
  end do
  deallocate(ws_pool)
  deallocate(u_cache, s_cache, uh_cache, sh_cache)
  deallocate(RTEdw_cache, RTEup_cache, RTMdw_cache, RTMup_cache, AdmInt_cache)
  deallocate(eta_shared)

end subroutine simulate

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! simulate_v8 — Interface f2py com suporte a F5 (freq arbitrárias) e F7 (tilted)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
!
! Extensão de simulate() que aceita flags F5/F7 e retorna respostas tilted.
! Quando use_tilted == 0 e use_arb_freq == 0, comportamento idêntico a simulate().
!
! F5 — Frequências arbitrárias:
!   use_arb_freq == 0: emite aviso se nf > 2 (guard, backward compat)
!   use_arb_freq == 1: nf ∈ [1, 16] validado, sem restrição
!
! F7 — Antenas inclinadas:
!   use_tilted == 0: cH_tilted_out preenchido com zeros (ignorar)
!   use_tilted == 1: calcula H_tilted(β, φ) a partir do tensor completo cH_out
!     Fórmula (receptor inclinado, transmissor axial ẑ):
!       H_tilted(β, φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]
!     onde Hxz = cH(:,3), Hyz = cH(:,6), Hzz = cH(:,9).
!
! Filtro Adaptativo (v9.0):
!   filter_type == 0 (default): Werthmuller 201pt
!   filter_type == 1: Kong 61pt (rápido, 3.3× speedup)
!   filter_type == 2: Anderson 801pt (máxima precisão)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine simulate_v8(nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, &
                       n, resist, esp, nmmax, &
                       use_arb_freq, use_tilted, n_tilted, n_tilted_sz, &
                       beta_tilt, phi_tilt, &
                       filter_type_in, &
                       zrho_out, cH_out, cH_tilted_out)
  implicit none
  ! Argumentos de entrada — modelo geológico e configuração EM
  integer, intent(in) :: nf, ntheta, nTR, n, nmmax
  real(dp), intent(in) :: freq(nf), theta(ntheta), dTR(nTR), resist(n,2), esp(n)
  real(dp), intent(in) :: h1, tj, p_med
  ! Argumentos de entrada — flags F5/F7
  ! n_tilted_sz: tamanho efetivo dos arrays tilted (≥ 1 para evitar zero-size).
  ! Quando F7 desabilitado, o caller deve passar n_tilted=0, n_tilted_sz=1,
  ! beta_tilt=[0.], phi_tilt=[0.] — os valores são ignorados pelo Fortran.
  integer, intent(in) :: use_arb_freq, use_tilted, n_tilted, n_tilted_sz
  real(dp), intent(in) :: beta_tilt(n_tilted_sz), phi_tilt(n_tilted_sz)
  ! Filtro Adaptativo — tipo de filtro de Hankel
  ! 0=Werthmuller 201pt (default), 1=Kong 61pt (rápido), 2=Anderson 801pt (preciso)
  integer, intent(in) :: filter_type_in
  ! Argumentos de saída
  !f2py intent(in) :: nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, n, resist, esp, nmmax
  !f2py intent(in) :: use_arb_freq, use_tilted, n_tilted, n_tilted_sz, beta_tilt, phi_tilt
  !f2py intent(in) :: filter_type_in
  !f2py intent(out) :: zrho_out, cH_out, cH_tilted_out
  !f2py depend(nf) :: freq
  !f2py depend(ntheta) :: theta
  !f2py depend(nTR) :: dTR
  !f2py depend(n) :: resist, esp
  !f2py depend(n_tilted_sz) :: beta_tilt, phi_tilt
  !f2py depend(nTR, ntheta, nmmax, nf) :: zrho_out, cH_out
  !f2py depend(nTR, ntheta, nmmax, nf, n_tilted_sz) :: cH_tilted_out
  real(dp), intent(out) :: zrho_out(nTR, ntheta, nmmax, nf, 3)
  complex(dp), intent(out) :: cH_out(nTR, ntheta, nmmax, nf, 9)
  complex(dp), intent(out) :: cH_tilted_out(nTR, ntheta, nmmax, nf, n_tilted_sz)

  ! Variáveis locais (idênticas a perfila1DanisoOMP)
  integer :: i, j, k, itr, it, ii, t, tid
  ! Filtro Adaptativo — npt_active determinado por filter_type_in em runtime
  integer :: npt_active
  real(dp) :: thetarad, del
  real(dp) :: z1, pz, posTR(6), ang, seno, coss, px, Lsen, Lcos
  real(dp) :: Tx_l, Ty_l, Tz_l, x_l, y_l, z_l, r_k, omega_i
  real(dp) :: beta_rad, phi_rad
  complex(dp) :: zeta_i
  character(5) :: dipolo
  integer, dimension(:), allocatable :: nmed
  real(dp), dimension(:), allocatable :: h, prof, krJ0J1_l, wJ0_l, wJ1_l
  real(dp), dimension(:,:), allocatable :: krwJ0J1, zrho, eta_shared
  complex(dp), dimension(:,:), allocatable :: cH_local
  real(dp), dimension(:,:,:), allocatable :: z_rho1
  complex(dp), dimension(:,:,:), allocatable :: c_H1
  integer :: maxthreads, num_threads_k, num_threads_j
  type(thread_workspace), allocatable :: ws_pool(:)
  complex(dp), allocatable :: u_cache(:,:,:,:), s_cache(:,:,:,:)
  complex(dp), allocatable :: uh_cache(:,:,:,:), sh_cache(:,:,:,:)
  complex(dp), allocatable :: RTEdw_cache(:,:,:,:), RTEup_cache(:,:,:,:)
  complex(dp), allocatable :: RTMdw_cache(:,:,:,:), RTMup_cache(:,:,:,:)
  complex(dp), allocatable :: AdmInt_cache(:,:,:,:)

  del = 1.d-6
  zrho_out = 0.d0
  cH_out = cmplx(0.d0, 0.d0, kind=dp)
  cH_tilted_out = cmplx(0.d0, 0.d0, kind=dp)

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F5 — Validação de frequências arbitrárias
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  if (use_arb_freq == 0 .and. nf > 2) then
    write(*,'(A,I0,A)') '[F5 AVISO] nf = ', nf, ' > 2 com use_arbitrary_freq desabilitado.'
  end if
  if (use_arb_freq == 1) then
    if (nf < 1 .or. nf > 16) then
      write(*,'(A,I0,A)') '[F5 ERRO] nf = ', nf, ' fora do intervalo [1, 16].'
      return
    end if
  end if

  allocate(nmed(ntheta))
  z1 = -h1
  do i = 1, ntheta
    thetarad = theta(i) * pi / 18.d1
    if (dabs(cos(thetarad)) > del) then
      pz = p_med * cos(thetarad)
      nmed(i) = ceiling(tj / pz)
    else
      nmed(i) = ceiling(tj / p_med) + 1
    end if
  end do

  ! Filtro Adaptativo — seleção do filtro de Hankel por filter_type_in
  select case (filter_type_in)
  case (1)
    npt_active = 61
    call J0J1Kong(npt_active, krJ0J1_l, wJ0_l, wJ1_l)
  case (2)
    npt_active = 801
    call J0J1And(krJ0J1_l, wJ0_l, wJ1_l)
  case default
    npt_active = 201
    call J0J1Wer(npt_active, krJ0J1_l, wJ0_l, wJ1_l)
  end select

  allocate(krwJ0J1(npt_active,3))
  do i = 1, npt_active
    krwJ0J1(i,:) = (/ krJ0J1_l(i), wJ0_l(i), wJ1_l(i) /)
  end do

  call sanitize_hprof_well(n, esp, h, prof)
  dipolo = 'hmdxy'

  ! Configuração OpenMP
  call omp_set_max_active_levels(2)
  maxthreads = omp_get_max_threads()
  num_threads_k = max(1, min(ntheta, maxthreads))
  num_threads_j = max(1, maxthreads / num_threads_k)

  ! Alocação workspace pool (Fase 3 + 3b)
  allocate(ws_pool(0:maxthreads-1))
  do t = 0, maxthreads-1
    allocate(ws_pool(t)%Tudw(npt_active, n), ws_pool(t)%Txdw(npt_active, n))
    allocate(ws_pool(t)%Tuup(npt_active, n), ws_pool(t)%Txup(npt_active, n))
    allocate(ws_pool(t)%TEdwz(npt_active, n), ws_pool(t)%TEupz(npt_active, n))
    allocate(ws_pool(t)%Mxdw(npt_active), ws_pool(t)%Mxup(npt_active))
    allocate(ws_pool(t)%Eudw(npt_active), ws_pool(t)%Euup(npt_active))
    allocate(ws_pool(t)%FEdwz(npt_active), ws_pool(t)%FEupz(npt_active))
  end do

  ! Caches Fase 4
  allocate(u_cache(npt_active, n, nf, ntheta), s_cache(npt_active, n, nf, ntheta))
  allocate(uh_cache(npt_active, n, nf, ntheta), sh_cache(npt_active, n, nf, ntheta))
  allocate(RTEdw_cache(npt_active, n, nf, ntheta), RTEup_cache(npt_active, n, nf, ntheta))
  allocate(RTMdw_cache(npt_active, n, nf, ntheta), RTMup_cache(npt_active, n, nf, ntheta))
  allocate(AdmInt_cache(npt_active, n, nf, ntheta))
  allocate(eta_shared(n, 2))
  do ii = 1, n
    eta_shared(ii, 1) = 1.d0 / resist(ii, 1)
    eta_shared(ii, 2) = 1.d0 / resist(ii, 2)
  end do

  allocate(zrho(nf, 3), cH_local(nf, 9))
  allocate(z_rho1(nmmax, nf, 3), c_H1(nmmax, nf, 9))
  zrho = 0.d0
  cH_local = cmplx(0.d0, 0.d0, kind=dp)
  z_rho1 = 0.d0
  c_H1 = cmplx(0.d0, 0.d0, kind=dp)

  ! Loop principal: multi-TR × theta × medidas (idêntico a perfila1DanisoOMP)
  do itr = 1, nTR
    !$omp parallel do schedule(dynamic) num_threads(num_threads_k) &
    !$omp&        if(ntheta > 1) &
    !$omp&        private(k,ang,seno,coss,px,pz,Lsen,Lcos,r_k,omega_i,zeta_i,ii) &
    !$omp&        firstprivate(z_rho1,c_H1)
    do k = 1, ntheta
      ang = theta(k) * pi / 18.d1
      seno = sin(ang)
      coss = cos(ang)
      px = p_med * seno
      pz = p_med * coss
      Lsen = dTR(itr) * seno
      Lcos = dTR(itr) * coss

      r_k = dTR(itr) * dabs(seno)
      do ii = 1, nf
        omega_i = 2.d0 * pi * freq(ii)
        zeta_i = cmplx(0.d0, 1.d0, kind=dp) * omega_i * mu
        call commonarraysMD(n, npt_active, r_k, krwJ0J1(:,1), zeta_i, h, eta_shared, &
                            u_cache(:,:,ii,k), s_cache(:,:,ii,k), &
                            uh_cache(:,:,ii,k), sh_cache(:,:,ii,k), &
                            RTEdw_cache(:,:,ii,k), RTEup_cache(:,:,ii,k), &
                            RTMdw_cache(:,:,ii,k), RTMup_cache(:,:,ii,k), &
                            AdmInt_cache(:,:,ii,k))
      end do

      !$omp parallel do schedule(guided, 16) &
      !$omp&        num_threads(merge(maxthreads, num_threads_j, ntheta == 1)) &
      !$omp&        default(shared) &
      !$omp&        private(j, x_l, y_l, z_l, Tx_l, Ty_l, Tz_l, posTR, tid) &
      !$omp&        firstprivate(zrho, cH_local)
      do j = 1, nmed(k)
        x_l  = 0.d0 + (j-1) * px - Lsen / 2
        y_l  = 0.d0
        z_l  = z1 + (j-1) * pz - Lcos / 2
        Tx_l = 0.d0 + (j-1) * px + Lsen / 2
        Ty_l = 0.d0
        Tz_l = z1 + (j-1) * pz + Lcos / 2
        posTR = (/Tx_l, Ty_l, Tz_l, x_l, y_l, z_l/)
        if (ntheta == 1) then
          tid = omp_get_thread_num()
        else
          tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
        end if
        call fieldsinfreqs_cached_ws(ws_pool(tid), ang, nf, freq, posTR, dipolo, npt_active, &
                                      krwJ0J1, n, h, prof, resist, eta_shared, &
                                      u_cache(:,:,:,k),  s_cache(:,:,:,k),    &
                                      uh_cache(:,:,:,k), sh_cache(:,:,:,k),   &
                                      RTEdw_cache(:,:,:,k), RTEup_cache(:,:,:,k), &
                                      RTMdw_cache(:,:,:,k), RTMup_cache(:,:,:,k), &
                                      AdmInt_cache(:,:,:,k),                  &
                                      zrho, cH_local)
        z_rho1(j,:,:) = zrho
        c_H1(j,:,:) = cH_local
      end do
      !$omp end parallel do
      zrho_out(itr, k, 1:nmed(k), :, :) = z_rho1(1:nmed(k), :, :)
      cH_out(itr, k, 1:nmed(k), :, :) = c_H1(1:nmed(k), :, :)
    end do
    !$omp end parallel do

    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! F7 — Cálculo das respostas de antenas inclinadas (pós-processamento)
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! Fórmula: H_tilted(β,φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]
    ! Mapeamento do tensor cH_out(:,:,:,:,1:9):
    !   cH(:,3)=Hxz, cH(:,6)=Hyz, cH(:,9)=Hzz
    if (use_tilted == 1 .and. n_tilted > 0) then
      do it = 1, n_tilted
        beta_rad = beta_tilt(it) * pi / 18.d1
        phi_rad  = phi_tilt(it) * pi / 18.d1
        do k = 1, ntheta
          do j = 1, nmed(k)
            do i = 1, nf
              cH_tilted_out(itr, k, j, i, it) = &
                cos(beta_rad) * cH_out(itr, k, j, i, 9) + &
                sin(beta_rad) * (cos(phi_rad) * cH_out(itr, k, j, i, 3) + &
                                 sin(phi_rad) * cH_out(itr, k, j, i, 6))
            end do
          end do
        end do
      end do
    end if
  end do

  ! Desalocação
  deallocate(zrho, cH_local, z_rho1, c_H1)
  if (allocated(krwJ0J1)) deallocate(krwJ0J1)
  do t = 0, maxthreads-1
    if (allocated(ws_pool(t)%Tudw)) deallocate(ws_pool(t)%Tudw)
    if (allocated(ws_pool(t)%Txdw)) deallocate(ws_pool(t)%Txdw)
    if (allocated(ws_pool(t)%Tuup)) deallocate(ws_pool(t)%Tuup)
    if (allocated(ws_pool(t)%Txup)) deallocate(ws_pool(t)%Txup)
    if (allocated(ws_pool(t)%TEdwz)) deallocate(ws_pool(t)%TEdwz)
    if (allocated(ws_pool(t)%TEupz)) deallocate(ws_pool(t)%TEupz)
    if (allocated(ws_pool(t)%Mxdw)) deallocate(ws_pool(t)%Mxdw)
    if (allocated(ws_pool(t)%Mxup)) deallocate(ws_pool(t)%Mxup)
    if (allocated(ws_pool(t)%Eudw)) deallocate(ws_pool(t)%Eudw)
    if (allocated(ws_pool(t)%Euup)) deallocate(ws_pool(t)%Euup)
    if (allocated(ws_pool(t)%FEdwz)) deallocate(ws_pool(t)%FEdwz)
    if (allocated(ws_pool(t)%FEupz)) deallocate(ws_pool(t)%FEupz)
  end do
  deallocate(ws_pool)
  deallocate(u_cache, s_cache, uh_cache, sh_cache)
  deallocate(RTEdw_cache, RTEup_cache, RTMdw_cache, RTMup_cache, AdmInt_cache)
  deallocate(eta_shared)

end subroutine simulate_v8

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! simulate_v10_jacobian — Wrapper f2py v10.0 com suporte a F10 (Jacobiano)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
!
! Extensão de simulate_v8 com saída adicional do Jacobiano ∂H/∂ρ calculado
! via Estratégia C (compute_jacobian_fd, OpenMP interno).
!
! Outputs adicionais (comparado a simulate_v8):
!   dH_dRho_h_out(nTR, ntheta, nmmax, nf, 9, n) complex(dp)
!   dH_dRho_v_out(nTR, ntheta, nmmax, nf, 9, n) complex(dp)
!
! Quando use_jacobian_in == 0: outputs preenchidos com zeros, sem custo
!                              computacional adicional (backward compat).
! Quando use_jacobian_in == 1: chama compute_jacobian_fd dentro do loop itr.
!
! Uso em Python:
!   import tatu_f2py
!   zrho, cH, cH_tilted, dJ_h, dJ_v = tatu_f2py.simulate_v10_jacobian(
!       nf=2, freq=np.array([20000., 40000.]),
!       ntheta=1, theta=np.array([0.]),
!       h1=10., tj=120., nTR=1, dTR=np.array([1.0]), p_med=0.2,
!       n=3, resist=np.array([[1.,1.],[10.,10.],[100.,100.]]),
!       esp=np.array([0.,50.,0.]), nmmax=600,
!       use_arb_freq=0, use_tilted=0, n_tilted=0, n_tilted_sz=1,
!       beta_tilt=np.zeros(1), phi_tilt=np.zeros(1),
!       filter_type_in=0, use_jacobian_in=1, jacobian_fd_step_in=1e-4)
!   # dJ_h.shape == (1, 1, 600, 2, 9, 3)
!
! Ref: docs/reference/relatorio_vantagens_jacobiano.md §9.3
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine simulate_v10_jacobian(nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, &
                                  n, resist, esp, nmmax,                             &
                                  use_arb_freq, use_tilted, n_tilted, n_tilted_sz,   &
                                  beta_tilt, phi_tilt,                               &
                                  filter_type_in,                                    &
                                  use_jacobian_in, jacobian_fd_step_in,              &
                                  zrho_out, cH_out, cH_tilted_out,                   &
                                  dH_dRho_h_out, dH_dRho_v_out)
  implicit none
  ! ── Entradas (modelo geológico + configuração EM) ──
  integer,  intent(in) :: nf, ntheta, nTR, n, nmmax
  real(dp), intent(in) :: freq(nf), theta(ntheta), dTR(nTR), resist(n,2), esp(n)
  real(dp), intent(in) :: h1, tj, p_med
  ! ── Entradas (flags F5/F7) ──
  integer,  intent(in) :: use_arb_freq, use_tilted, n_tilted, n_tilted_sz
  real(dp), intent(in) :: beta_tilt(n_tilted_sz), phi_tilt(n_tilted_sz)
  ! ── Entradas (Filtro Adaptativo) ──
  integer,  intent(in) :: filter_type_in
  ! ── Entradas (F10 — Jacobiano) ──
  integer,  intent(in) :: use_jacobian_in        ! 0=off, 1=on
  real(dp), intent(in) :: jacobian_fd_step_in    ! default 1e-4 relativo
  ! ── Diretivas f2py ──
  !f2py intent(in) :: nf, freq, ntheta, theta, h1, tj, nTR, dTR, p_med, n, resist, esp, nmmax
  !f2py intent(in) :: use_arb_freq, use_tilted, n_tilted, n_tilted_sz, beta_tilt, phi_tilt
  !f2py intent(in) :: filter_type_in, use_jacobian_in, jacobian_fd_step_in
  !f2py intent(out) :: zrho_out, cH_out, cH_tilted_out, dH_dRho_h_out, dH_dRho_v_out
  !f2py depend(nf) :: freq
  !f2py depend(ntheta) :: theta
  !f2py depend(nTR) :: dTR
  !f2py depend(n) :: resist, esp
  !f2py depend(n_tilted_sz) :: beta_tilt, phi_tilt
  !f2py depend(nTR, ntheta, nmmax, nf) :: zrho_out, cH_out
  !f2py depend(nTR, ntheta, nmmax, nf, n_tilted_sz) :: cH_tilted_out
  !f2py depend(nTR, ntheta, nmmax, nf, n) :: dH_dRho_h_out, dH_dRho_v_out
  real(dp),    intent(out) :: zrho_out(nTR, ntheta, nmmax, nf, 3)
  complex(dp), intent(out) :: cH_out(nTR, ntheta, nmmax, nf, 9)
  complex(dp), intent(out) :: cH_tilted_out(nTR, ntheta, nmmax, nf, n_tilted_sz)
  complex(dp), intent(out) :: dH_dRho_h_out(nTR, ntheta, nmmax, nf, 9, n)
  complex(dp), intent(out) :: dH_dRho_v_out(nTR, ntheta, nmmax, nf, 9, n)

  ! ── Variáveis locais (idênticas a simulate_v8 + F10) ──
  integer :: i, j, k, itr, it, ii, t, tid, jj
  integer :: npt_active
  real(dp) :: thetarad, del
  real(dp) :: z1, pz, posTR(6), ang, seno, coss, px, Lsen, Lcos
  real(dp) :: Tx_l, Ty_l, Tz_l, x_l, y_l, z_l, r_k, omega_i
  real(dp) :: beta_rad, phi_rad
  complex(dp) :: zeta_i
  character(5) :: dipolo
  integer,  dimension(:),     allocatable :: nmed
  real(dp), dimension(:),     allocatable :: h, prof, krJ0J1_l, wJ0_l, wJ1_l
  real(dp), dimension(:,:),   allocatable :: krwJ0J1, zrho, eta_shared
  complex(dp), dimension(:,:), allocatable :: cH_local
  real(dp), dimension(:,:,:),  allocatable :: z_rho1
  complex(dp), dimension(:,:,:), allocatable :: c_H1
  integer :: maxthreads, num_threads_k, num_threads_j
  type(thread_workspace), allocatable :: ws_pool(:)
  complex(dp), allocatable :: u_cache(:,:,:,:), s_cache(:,:,:,:)
  complex(dp), allocatable :: uh_cache(:,:,:,:), sh_cache(:,:,:,:)
  complex(dp), allocatable :: RTEdw_cache(:,:,:,:), RTEup_cache(:,:,:,:)
  complex(dp), allocatable :: RTMdw_cache(:,:,:,:), RTMup_cache(:,:,:,:)
  complex(dp), allocatable :: AdmInt_cache(:,:,:,:)
  ! F10 — posições por ângulo para compute_jacobian_fd
  real(dp), allocatable :: posTR_array_k(:,:)

  del = 1.d-6
  zrho_out      = 0.d0
  cH_out        = cmplx(0.d0, 0.d0, kind=dp)
  cH_tilted_out = cmplx(0.d0, 0.d0, kind=dp)
  dH_dRho_h_out = cmplx(0.d0, 0.d0, kind=dp)
  dH_dRho_v_out = cmplx(0.d0, 0.d0, kind=dp)

  ! ── Validação F5 (nf arbitrário) ──
  if (use_arb_freq == 0 .and. nf > 2) then
    write(*,'(A,I0,A)') '[F5 AVISO] nf = ', nf, ' > 2 com use_arbitrary_freq desabilitado.'
  end if
  if (use_arb_freq == 1) then
    if (nf < 1 .or. nf > 16) then
      write(*,'(A,I0,A)') '[F5 ERRO] nf = ', nf, ' fora do intervalo [1, 16].'
      return
    end if
  end if

  ! ── Seleção do Filtro Adaptativo ──
  select case (filter_type_in)
  case (1)
    npt_active = 61
    call J0J1Kong(npt_active, krJ0J1_l, wJ0_l, wJ1_l)
  case (2)
    npt_active = 801
    call J0J1And(krJ0J1_l, wJ0_l, wJ1_l)
  case default
    npt_active = 201
    call J0J1Wer(npt_active, krJ0J1_l, wJ0_l, wJ1_l)
  end select

  allocate(nmed(ntheta))
  z1 = -h1
  do i = 1, ntheta
    thetarad = theta(i) * pi / 18.d1
    if (dabs(cos(thetarad)) > del) then
      pz = p_med * cos(thetarad)
      nmed(i) = ceiling(tj / pz)
    else
      nmed(i) = ceiling(tj / p_med) + 1
    end if
  end do

  allocate(krwJ0J1(npt_active, 3))
  do i = 1, npt_active
    krwJ0J1(i, :) = (/ krJ0J1_l(i), wJ0_l(i), wJ1_l(i) /)
  end do

  call sanitize_hprof_well(n, esp, h, prof)
  dipolo = 'hmdxy'

  ! ── Configuração OpenMP ──
  call omp_set_max_active_levels(2)
  maxthreads = omp_get_max_threads()
  num_threads_k = max(1, min(ntheta, maxthreads))
  num_threads_j = max(1, maxthreads / num_threads_k)

  ! ── Alocação workspace pool ──
  allocate(ws_pool(0:maxthreads-1))
  do t = 0, maxthreads-1
    allocate(ws_pool(t)%Tudw(npt_active, n), ws_pool(t)%Txdw(npt_active, n))
    allocate(ws_pool(t)%Tuup(npt_active, n), ws_pool(t)%Txup(npt_active, n))
    allocate(ws_pool(t)%TEdwz(npt_active, n), ws_pool(t)%TEupz(npt_active, n))
    allocate(ws_pool(t)%Mxdw(npt_active), ws_pool(t)%Mxup(npt_active))
    allocate(ws_pool(t)%Eudw(npt_active), ws_pool(t)%Euup(npt_active))
    allocate(ws_pool(t)%FEdwz(npt_active), ws_pool(t)%FEupz(npt_active))
  end do

  ! ── Alocação dos caches Phase 4 ──
  allocate(u_cache(npt_active, n, nf, ntheta), s_cache(npt_active, n, nf, ntheta))
  allocate(uh_cache(npt_active, n, nf, ntheta), sh_cache(npt_active, n, nf, ntheta))
  allocate(RTEdw_cache(npt_active, n, nf, ntheta), RTEup_cache(npt_active, n, nf, ntheta))
  allocate(RTMdw_cache(npt_active, n, nf, ntheta), RTMup_cache(npt_active, n, nf, ntheta))
  allocate(AdmInt_cache(npt_active, n, nf, ntheta))
  allocate(eta_shared(n, 2))
  do ii = 1, n
    eta_shared(ii, 1) = 1.d0 / resist(ii, 1)
    eta_shared(ii, 2) = 1.d0 / resist(ii, 2)
  end do

  allocate(zrho(nf, 3), cH_local(nf, 9))
  allocate(z_rho1(nmmax, nf, 3), c_H1(nmmax, nf, 9))
  zrho = 0.d0
  cH_local = cmplx(0.d0, 0.d0, kind=dp)

  ! ── Loop principal multi-TR × theta × medidas ──
  do itr = 1, nTR
    !$omp parallel do schedule(dynamic) num_threads(num_threads_k) &
    !$omp&        if(ntheta > 1) &
    !$omp&        private(k,ang,seno,coss,px,pz,Lsen,Lcos,r_k,omega_i,zeta_i,ii) &
    !$omp&        firstprivate(z_rho1,c_H1)
    do k = 1, ntheta
      ang = theta(k) * pi / 18.d1
      seno = sin(ang)
      coss = cos(ang)
      px = p_med * seno
      pz = p_med * coss
      Lsen = dTR(itr) * seno
      Lcos = dTR(itr) * coss

      r_k = dTR(itr) * dabs(seno)
      do ii = 1, nf
        omega_i = 2.d0 * pi * freq(ii)
        zeta_i = cmplx(0.d0, 1.d0, kind=dp) * omega_i * mu
        call commonarraysMD(n, npt_active, r_k, krwJ0J1(:,1), zeta_i, h, eta_shared, &
                            u_cache(:,:,ii,k), s_cache(:,:,ii,k), &
                            uh_cache(:,:,ii,k), sh_cache(:,:,ii,k), &
                            RTEdw_cache(:,:,ii,k), RTEup_cache(:,:,ii,k), &
                            RTMdw_cache(:,:,ii,k), RTMup_cache(:,:,ii,k), &
                            AdmInt_cache(:,:,ii,k))
      end do

      !$omp parallel do schedule(guided, 16) &
      !$omp&        num_threads(merge(maxthreads, num_threads_j, ntheta == 1)) &
      !$omp&        default(shared) &
      !$omp&        private(j, x_l, y_l, z_l, Tx_l, Ty_l, Tz_l, posTR, tid) &
      !$omp&        firstprivate(zrho, cH_local)
      do j = 1, nmed(k)
        x_l  = 0.d0 + (j-1) * px - Lsen / 2
        y_l  = 0.d0
        z_l  = z1 + (j-1) * pz - Lcos / 2
        Tx_l = 0.d0 + (j-1) * px + Lsen / 2
        Ty_l = 0.d0
        Tz_l = z1 + (j-1) * pz + Lcos / 2
        posTR = (/Tx_l, Ty_l, Tz_l, x_l, y_l, z_l/)
        if (ntheta == 1) then
          tid = omp_get_thread_num()
        else
          tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
        end if
        call fieldsinfreqs_cached_ws(ws_pool(tid), ang, nf, freq, posTR, dipolo, npt_active, &
                                      krwJ0J1, n, h, prof, resist, eta_shared, &
                                      u_cache(:,:,:,k),  s_cache(:,:,:,k),    &
                                      uh_cache(:,:,:,k), sh_cache(:,:,:,k),   &
                                      RTEdw_cache(:,:,:,k), RTEup_cache(:,:,:,k), &
                                      RTMdw_cache(:,:,:,k), RTMup_cache(:,:,:,k), &
                                      AdmInt_cache(:,:,:,k),                  &
                                      zrho, cH_local)
        z_rho1(j,:,:) = zrho
        c_H1(j,:,:) = cH_local
      end do
      !$omp end parallel do
      zrho_out(itr, k, 1:nmed(k), :, :) = z_rho1(1:nmed(k), :, :)
      cH_out(itr, k, 1:nmed(k), :, :) = c_H1(1:nmed(k), :, :)
    end do
    !$omp end parallel do

    ! ── F7: respostas tilted ──
    if (use_tilted == 1 .and. n_tilted > 0) then
      do it = 1, n_tilted
        beta_rad = beta_tilt(it) * pi / 18.d1
        phi_rad  = phi_tilt(it) * pi / 18.d1
        do k = 1, ntheta
          do j = 1, nmed(k)
            do i = 1, nf
              cH_tilted_out(itr, k, j, i, it) = &
                cos(beta_rad) * cH_out(itr, k, j, i, 9) + &
                sin(beta_rad) * (cos(phi_rad) * cH_out(itr, k, j, i, 3) + &
                                 sin(phi_rad) * cH_out(itr, k, j, i, 6))
            end do
          end do
        end do
      end do
    end if

    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! F10 — Cálculo do Jacobiano (Estratégia C, compute_jacobian_fd)
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    if (use_jacobian_in == 1) then
      do k = 1, ntheta
        ang  = theta(k) * pi / 18.d1
        seno = sin(ang)
        coss = cos(ang)
        px   = p_med * seno
        pz   = p_med * coss
        Lsen = dTR(itr) * seno
        Lcos = dTR(itr) * coss
        r_k  = dTR(itr) * dabs(seno)

        if (allocated(posTR_array_k)) deallocate(posTR_array_k)
        allocate(posTR_array_k(6, nmed(k)))
        do jj = 1, nmed(k)
          x_l  = 0.d0 + (jj-1) * px - Lsen / 2
          y_l  = 0.d0
          z_l  = z1 + (jj-1) * pz - Lcos / 2
          Tx_l = 0.d0 + (jj-1) * px + Lsen / 2
          Ty_l = 0.d0
          Tz_l = z1 + (jj-1) * pz + Lcos / 2
          posTR_array_k(:, jj) = (/ Tx_l, Ty_l, Tz_l, x_l, y_l, z_l /)
        end do

        call compute_jacobian_fd(ws_pool, maxthreads, ang, nf, freq,            &
                                  posTR_array_k, nmed(k), dipolo, npt_active,   &
                                  krwJ0J1, n, h, prof, resist, eta_shared, r_k, &
                                  jacobian_fd_step_in,                          &
                                  dH_dRho_h_out(itr, k, 1:nmed(k), :, :, :),    &
                                  dH_dRho_v_out(itr, k, 1:nmed(k), :, :, :))
      end do
    end if
  end do

  ! ── Desalocação ──
  if (allocated(posTR_array_k)) deallocate(posTR_array_k)
  deallocate(zrho, cH_local, z_rho1, c_H1)
  if (allocated(krwJ0J1)) deallocate(krwJ0J1)
  do t = 0, maxthreads-1
    if (allocated(ws_pool(t)%Tudw)) deallocate(ws_pool(t)%Tudw)
    if (allocated(ws_pool(t)%Txdw)) deallocate(ws_pool(t)%Txdw)
    if (allocated(ws_pool(t)%Tuup)) deallocate(ws_pool(t)%Tuup)
    if (allocated(ws_pool(t)%Txup)) deallocate(ws_pool(t)%Txup)
    if (allocated(ws_pool(t)%TEdwz)) deallocate(ws_pool(t)%TEdwz)
    if (allocated(ws_pool(t)%TEupz)) deallocate(ws_pool(t)%TEupz)
    if (allocated(ws_pool(t)%Mxdw)) deallocate(ws_pool(t)%Mxdw)
    if (allocated(ws_pool(t)%Mxup)) deallocate(ws_pool(t)%Mxup)
    if (allocated(ws_pool(t)%Eudw)) deallocate(ws_pool(t)%Eudw)
    if (allocated(ws_pool(t)%Euup)) deallocate(ws_pool(t)%Euup)
    if (allocated(ws_pool(t)%FEdwz)) deallocate(ws_pool(t)%FEdwz)
    if (allocated(ws_pool(t)%FEupz)) deallocate(ws_pool(t)%FEupz)
  end do
  deallocate(ws_pool)
  deallocate(u_cache, s_cache, uh_cache, sh_cache)
  deallocate(RTEdw_cache, RTEup_cache, RTMdw_cache, RTMup_cache, AdmInt_cache)
  deallocate(eta_shared)
  deallocate(nmed)

end subroutine simulate_v10_jacobian

end module tatu_wrapper
