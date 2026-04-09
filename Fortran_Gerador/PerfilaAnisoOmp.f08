module DManisoTIV
use parameters
use filterscommonbase
use utils
use magneticdipoles
use omp_lib
contains
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, &
                             nTR, dTR, p_med, n, resist, esp, filename, &
                             use_arb_freq, use_tilted, n_tilted, beta_tilt, phi_tilt, &
                             use_compensation, n_comp_pairs, comp_pairs, &
                             filter_type)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Sub-rotina principal do simulador EM 1D TIV com múltiplos pares T-R.
  !
  ! Versão 9.0: F5 + F7 + F6 (compensação midpoint) + Filtro Adaptativo.
  !   - nTR = 1: backward-compatible (saída idêntica ao formato anterior)
  !   - nTR > 1: loop externo sobre pares T-R, cada um com r_k = dTR(itr)*|sin(θ)|
  !              Saída: arquivos separados por par T-R (sufixo _TR{itr})
  !
  ! F5 — Frequências arbitrárias (use_arb_freq):
  !   Quando use_arb_freq == 0 (default): emite aviso se nf > 2 (guard)
  !   Quando use_arb_freq == 1: nf arbitrário (1-16), sem restrição
  !   O código já suporta nf arbitrário via caches Phase 4: (npt, n, nf, ntheta)
  !
  ! F6 — Compensação midpoint (use_compensation):
  !   Quando use_compensation == 0 (default): sem compensação, saída padrão
  !   Quando use_compensation == 1: calcula medições compensadas para pares T-R:
  !     - Diferença de fase: Δφ = arg(H_near) − arg(H_far)
  !     - Atenuação: Δα = 20·log₁₀(|H_near|/|H_far|)
  !     - Tensor compensado: H_comp = (H_near + H_far) / 2
  !   n_comp_pairs: número de pares de compensação
  !   comp_pairs(n_comp_pairs, 2): índices (near_itr, far_itr) dos pares T-R
  !   Requer nTR ≥ 2. Cada par gera arquivo _COMP{ipair}.dat adicional.
  !   Ref: docs/reference/analise_novos_recursos_simulador_fortran.md §5
  !
  ! F7 — Antenas inclinadas (use_tilted):
  !   Quando use_tilted == 0 (default): sem cálculo extra, saída inalterada (22 col)
  !   Quando use_tilted == 1: calcula H_tilted para cada configuração (β, φ):
  !     H_tilted(β, φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]
  !   onde β = ângulo de inclinação (0°-90°), φ = azimute (0°-360°).
  !   Saída estendida: 22 + 2×n_tilted colunas por registro.
  !   Ref: docs/reference/analise_novos_recursos_simulador_fortran.md §F7
  !
  ! Filtro Adaptativo (filter_type):
  !   filter_type == 0 (default): Werthmuller 201 pontos (precisão 10⁻⁶)
  !   filter_type == 1: Kong 61 pontos (rápido, 3.3×, precisão 10⁻⁴)
  !   filter_type == 2: Anderson 801 pontos (máxima precisão 10⁻⁸, 4× lento)
  !   Seleção no model.in controla trade-off velocidade × precisão.
  !   Kong recomendado para geração de datasets de treinamento (ruído será
  !   adicionado); Anderson para validação cruzada com empymod.
  !   Ref: docs/reference/analise_novos_recursos_simulador_fortran.md §7.2.4
  !
  ! Fluxo de execução (multi-TR):
  !   do itr = 1, nTR
  !     do k = 1, ntheta  (outer parallel if ntheta > 1)
  !       r_k = dTR(itr) * |sin(θ_k)|
  !       commonarraysMD → cache(npt, n, nf)  [1× por (itr, k)]
  !       do j = 1, nmed(k)  (inner parallel, schedule(guided, 16))
  !         fieldsinfreqs_cached_ws(ws_pool(tid), cache, ...)
  !       end do
  !     end do
  !     [F7: compute tilted responses from cH1 tensor]
  !     writes_files(...)
  !   end do
  !   [F6: compute compensation from stored multi-TR data]
  !   borehole_compensation(cH_all_tr, comp_pairs, ...)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  implicit none
  character(*), intent(in) :: mypath
  integer, intent(in) ::  modelm, nmaxmodel, nf, ntheta, n, nTR
  real(dp), intent(in) :: freq(nf), theta(ntheta), h1, tj, dTR(nTR), p_med, resist(n,2), esp(n)
  character(*), intent(in) :: filename
  ! F5/F7 — Feature flags (desabilitados por padrão = 0, backward compatible)
  integer, intent(in) :: use_arb_freq   ! F5: 0=padrão (guard nf>2), 1=nf arbitrário
  integer, intent(in) :: use_tilted     ! F7: 0=desabilitado, 1=calcula antenas inclinadas
  integer, intent(in) :: n_tilted       ! F7: número de configurações tilted (0 se desabilitado)
  real(dp), intent(in) :: beta_tilt(:)  ! F7: ângulos de inclinação em graus (size n_tilted)
  real(dp), intent(in) :: phi_tilt(:)   ! F7: ângulos azimutais em graus (size n_tilted)
  ! F6 — Compensação midpoint (borehole compensation)
  integer, intent(in) :: use_compensation  ! F6: 0=desabilitado (default), 1=habilitado
  integer, intent(in) :: n_comp_pairs      ! F6: número de pares de compensação (0 se desab.)
  integer, intent(in) :: comp_pairs(:,:)   ! F6: pares (near_itr, far_itr), shape (n_comp_pairs, 2)
  ! Filtro Adaptativo — seleção do filtro de Hankel
  integer, intent(in) :: filter_type  ! 0=Werthmuller 201pt (default), 1=Kong 61pt, 2=Anderson 801pt

  integer :: i, j, k, nmmax
  real(dp) :: thetamin, thetaplu, thetarad
  ! real(dp) :: tj
  character(5) :: dipolo
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Filtro Adaptativo — npt_active determinado por filter_type em runtime.
  !   filter_type == 0: Werthmuller 201pt (default, precisão 10⁻⁶)
  !   filter_type == 1: Kong 61pt (rápido, precisão 10⁻⁴, 3.3× speedup)
  !   filter_type == 2: Anderson 801pt (máxima precisão 10⁻⁸, 4× mais lento)
  !
  ! PERFORMANCE: npt_active é variável runtime (não parameter) porque o filtro
  ! é selecionável pelo usuário. Todas as alocações de workspace, caches e
  ! arrays de filtro usam npt_active. As sub-rotinas fieldsinfreqs/fieldsinfreqs_ws
  ! (legado, NÃO chamadas no hot path Phase 4) recebem npt como dummy argument
  ! com intent(in) — inalteradas.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  integer :: npt_active
  real(dp), dimension(:), allocatable :: krJ0J1, wJ0, wJ1
  real(dp), dimension(:,:), allocatable :: krwJ0J1
  !=============================================================================
  real(dp) :: z1, pz, posTR(6), ang, seno, coss, px, Lsen, Lcos !, zn
  real(dp) :: Tx, Ty, Tz, x, y, z !, ACx1, ACp1, ACx2, ACp2, ACx3, ACp3
  integer, dimension(:), allocatable :: nmed
  real(dp), dimension(:), allocatable :: h, prof
  !=============================================================================
  real(dp), dimension(:,:), allocatable :: zrho
  complex(dp), dimension(:,:), allocatable :: cH
  real(dp), dimension(:,:,:), allocatable :: z_rho1
  complex(dp), dimension(:,:,:), allocatable :: c_H1
  real(dp), dimension(:,:,:,:), allocatable :: zrho1
  complex(dp), dimension(:,:,:,:), allocatable :: cH1
  integer :: maxthreads, num_threads_k, num_threads_j !, nrows
  logical :: nested_enabled
  !=============================================================================
  ! Fase 3 — Workspace pool pré-alocado por thread (ver magneticdipoles.f08)
  ! Elimina todas as chamadas allocate/deallocate dentro do loop paralelo de
  ! hmd_TIV_optimized e vmd_optimized, removendo a contenção no mutex do heap.
  ! Ref: docs/reference/analise_paralelismo_cpu_fortran.md §7 Fase 3
  !=============================================================================
  type(thread_workspace), allocatable :: ws_pool(:)
  integer :: t, tid, itr

  !=============================================================================
  ! Fase 4 — Cache de commonarraysMD por (r, freq)
  !
  ! Pré-computa o resultado de commonarraysMD uma única vez por ângulo/frequência
  ! em vez de chamá-la nf × nmed = 1.200 vezes por modelo. A redundância é 100%
  ! porque commonarraysMD(n, npt, r, freq, h, eta, ...) depende apenas de
  ! (r, freq, n, h, eta) — invariantes em j dentro de um modelo.
  !
  ! Os 9 arrays abaixo são alocados no heap uma única vez por modelo (em
  ! perfila1DanisoOMP, fora de qualquer loop), preenchidos no início de cada
  ! iteração k, e lidos pelos threads no inner parallel via shared clause.
  !
  ! Tamanho: 9 × (npt × n × nf) × 16 bytes ≈ 1,68 MB para n=29, nf=2, npt=201.
  ! Ref: docs/reference/analise_paralelismo_cpu_fortran.md §7 Fase 4
  !=============================================================================
  ! Fase 4 — cache com dimensão ntheta: u_cache(npt, n, nf, ntheta).
  ! Necessário porque o outer !$omp parallel do if(ntheta>1) executa k=1 e k=2
  ! em threads distintos. Sem a dimensão ntheta, thread k=1 e thread k=2
  ! escreviam no mesmo u_cache(:,:,ii) simultaneamente → race condition que
  ! corromperia silenciosamente os resultados de θ=30°.
  ! Com a 4ª dimensão cada thread k escreve em u_cache(:,:,ii,k) independente.
  complex(dp), allocatable :: u_cache(:,:,:,:), s_cache(:,:,:,:)
  complex(dp), allocatable :: uh_cache(:,:,:,:), sh_cache(:,:,:,:)
  complex(dp), allocatable :: RTEdw_cache(:,:,:,:), RTEup_cache(:,:,:,:)
  complex(dp), allocatable :: RTMdw_cache(:,:,:,:), RTMup_cache(:,:,:,:)
  complex(dp), allocatable :: AdmInt_cache(:,:,:,:)
  real(dp)    :: r_k, omega_i
  complex(dp) :: zeta_i
  real(dp), allocatable :: eta_shared(:,:)  ! (n, 2) — hoisted de fieldsinfreqs (B2)
  integer     :: ii
  ! real(dp) :: wtime

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F7 — Variáveis para antenas inclinadas (tilted coils)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! A resposta de uma antena inclinada com eixo n̂ = (sinβ·cosφ, sinβ·sinφ, cosβ)
  ! medindo o campo de um transmissor axial (ẑ) é:
  !   H_tilted(β, φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]
  ! onde Hxz = cH(:,3), Hyz = cH(:,6), Hzz = cH(:,9) do tensor 3×3.
  ! Custo: 5 multiplicações + 2 adições por ponto — negligível vs forward model.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  complex(dp), allocatable :: cH_tilted(:,:,:,:)  ! (ntheta, nmmax, nf, n_tilted)
  real(dp) :: beta_rad, phi_rad
  integer  :: it

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F6 — Variáveis para compensação midpoint (borehole compensation)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Armazena o tensor completo por par T-R para pós-processamento de compensação.
  ! cH_all_tr(nTR, ntheta, nmmax, nf, 9): tensor H para todos os pares T-R.
  !   Preenchido durante o loop do itr = 1, nTR.
  !   Usado após o loop para calcular medições compensadas (phase_diff, atten).
  ! zrho_all_tr(nTR, ntheta, nmmax, nf, 3): zobs, rho_h, rho_v por par T-R.
  ! cH_comp(n_comp_pairs, ntheta, nmmax, nf, 9): tensor compensado por par.
  ! phase_diff(n_comp_pairs, ntheta, nmmax, nf, 9): diferença de fase (graus).
  ! atten_db(n_comp_pairs, ntheta, nmmax, nf, 9): atenuação (dB).
  !
  ! Memória: para nTR=3, ntheta=1, nmmax=600, nf=2:
  !   cH_all_tr: 3 × 1 × 600 × 2 × 9 × 16 bytes = ~518 KB — negligível.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  complex(dp), allocatable :: cH_all_tr(:,:,:,:,:)    ! (nTR, ntheta, nmmax, nf, 9)
  real(dp), allocatable    :: zrho_all_tr(:,:,:,:,:)   ! (nTR, ntheta, nmmax, nf, 3)
  complex(dp), allocatable :: cH_comp(:,:,:,:,:)       ! (n_comp_pairs, ntheta, nmmax, nf, 9)
  real(dp), allocatable    :: phase_diff(:,:,:,:,:)    ! (n_comp_pairs, ntheta, nmmax, nf, 9)
  real(dp), allocatable    :: atten_db(:,:,:,:,:)      ! (n_comp_pairs, ntheta, nmmax, nf, 9)
  integer :: ipair, i_near, i_far, ic
  real(dp) :: abs_near, abs_far

  ! wtime = omp_get_wtime( )

  ! nrows = 0 !número de linhas do arquivo de saída
  allocate(nmed(ntheta))
  z1 = - h1
  ! zn = sum(esp(2:(n-1))) + hn
  ! tj = zn - z1  !comprimento da janela, vertical, entre a primeira posição e a última posição dos ponto-médios de medidas
  do i = 1,ntheta
    thetamin = theta(i) - del
    thetaplu = theta(i) + del
    if ( (thetamin > 0.d0 .and. thetaplu < 9.d1) .or. (thetamin < 0.d0 .and. thetaplu > 0.d0) ) then
      thetarad = theta(i) * pi / 18.d1
      pz = p_med * cos(thetarad)
      nmed(i) = ceiling(tj / pz)  !idnint(tj / pz) + 1
      ! write(*,*)'Número de medidas',nmed(i)
    elseif ( thetamin <= 9.d1 .and. thetaplu >= 9.d1 .and. dabs(tj) > del ) then !caso de investigação horizontal
      write(*,*)'Existe uma perfilagem horizontal'
      nmed(i) = ceiling(tj / p_med) + 1 !dedicando o número de medidas sendo o mesmo de uma investigação vertical
    else
      stop 'Apenas ângulos entre 0 e 90 graus são admitidos quando as posições primeira e última de medida são distintas!'
    end if
    ! nrows = nrows + nmed(i) * nf  !número de linhas do arquivo de saída
  end do
  nmmax = maxval(nmed)

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Filtro Adaptativo — Seleção do filtro de Hankel por filter_type
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! O filtro de Hankel discretiza a integral EM:
  !   H(r,z,ω) = ∫₀^∞ K(kr,z,ω) · J_ν(kr·r) · kr dkr
  ! usando npt pesos pré-computados (quadratura digital).
  !
  ! Seleção por cenário:
  !   ┌────────────────────┬───────────────────┬──────┬────────────┐
  !   │ filter_type        │ Filtro            │ npt  │ Precisão   │
  !   ├────────────────────┼───────────────────┼──────┼────────────┤
  !   │ 0 (default)        │ Werthmuller       │ 201  │ 10⁻⁶       │
  !   │ 1 (rápido)         │ Kong              │  61  │ 10⁻⁴       │
  !   │ 2 (máxima prec.)   │ Anderson          │ 801  │ 10⁻⁸       │
  !   └────────────────────┴───────────────────┴──────┴────────────┘
  !
  ! Custo computacional: escala linearmente com npt.
  !   Kong (61): ~3.3× mais rápido que Werthmuller (201)
  !   Anderson (801): ~4× mais lento que Werthmuller (201)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  select case (filter_type)
  case (1)
    ! Kong 61 pontos — geração rápida de datasets de treinamento
    npt_active = 61
    call J0J1Kong(npt_active, krJ0J1, wJ0, wJ1)
  case (2)
    ! Anderson 801 pontos — validação e referência (máxima precisão)
    npt_active = 801
    call J0J1And(krJ0J1, wJ0, wJ1)
  case default
    ! Werthmuller 201 pontos — simulação padrão (backward compatible)
    npt_active = 201
    call J0J1Wer(npt_active, krJ0J1, wJ0, wJ1)
  end select

  allocate(krwJ0J1(npt_active,3))
  do i = 1, npt_active
    krwJ0J1(i,:) = (/ krJ0J1(i), wJ0(i), wJ1(i) /)
  end do

  call sanitize_hprof_well(n, esp, h, prof)
  dipolo = 'hmdxy'  !'hmdx' !'hmdy' !variável que especifica que dipolo magnético horizontal se deseja simular

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Fase 2 — Hybrid Scheduler + Correção Débito 2 (OpenMP moderno)
  ! Ref: docs/reference/analise_paralelismo_cpu_fortran.md §7 Fase 2
  !       docs/reference/relatorio_fase0_fase1_fortran.md §4.2, §4.4
  !
  ! Débito 2 corrigido: omp_set_nested/omp_get_nested são APIs depreciadas
  ! desde OpenMP 5.0 (2018) e serão removidas em versões futuras. A API
  ! substituta é omp_set_max_active_levels(n), que controla diretamente
  ! quantos níveis de paralelismo aninhado podem estar ativos.
  !
  ! Fase 2 + Débito 3 (particionamento de threads): o código original usava
  ! subtração (num_threads_j = maxthreads - ntheta), que degenera o loop
  ! interno a 1 thread quando OMP_NUM_THREADS=2 e ntheta=1, causando a
  ! anti-escalabilidade observada em Fase 0 (speedup 0,94× em 2 threads).
  ! A nova lógica usa particionamento MULTIPLICATIVO:
  !   num_threads_k × num_threads_j ≈ maxthreads
  !
  ! Tabela de distribuição (validada empiricamente na Fase 0):
  !   ┌──────────┬──────────┬──────────┬──────────┬─────────────────────┐
  !   │ maxthr   │  ntheta  │   n_k    │   n_j    │  n_k × n_j          │
  !   ├──────────┼──────────┼──────────┼──────────┼─────────────────────┤
  !   │    2     │    1     │    1     │    2     │   2   ✓ (fix bug)   │
  !   │    2     │    2     │    2     │    1     │   2                 │
  !   │    8     │    1     │    1     │    8     │   8   ✓             │
  !   │    8     │    2     │    2     │    4     │   8                 │
  !   │    8     │    7     │    7     │    1     │   7 (~maxthr)       │
  !   │   16     │    1     │    1     │   16     │  16   ✓             │
  !   │   16     │    7     │    7     │    2     │  14 (~maxthr)       │
  !   └──────────┴──────────┴──────────┴──────────┴─────────────────────┘
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  call omp_set_max_active_levels(2)
  nested_enabled = (omp_get_max_active_levels() >= 2)

  ! Particionamento MULTIPLICATIVO de threads (Fase 2 + Débito 3)
  ! num_threads_k: limitado por ntheta (não há mais trabalho do que ângulos)
  ! num_threads_j: distribui threads restantes no nível interno (medidas)
  maxthreads = omp_get_max_threads()
  num_threads_k = max(1, min(ntheta, maxthreads))
  num_threads_j = max(1, maxthreads / num_threads_k)

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Fase 3 — Alocação do ws_pool (Workspace Pre-allocation)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Aloca um workspace exclusivo por thread ANTES do loop paralelo. Cada
  ! thread acessa ws_pool(tid) dentro do inner parallel via cálculo de um
  ! tid GLOBAL (ver D6 abaixo), eliminando a necessidade de allocate/deallocate
  ! dentro de hmd_TIV_optimized_ws e vmd_optimized_ws.
  !
  ! Débito 6 (RESOLVIDO em PR1-Hygiene pós-Fase 3):
  !   omp_get_thread_num() dentro do inner team retorna o tid do time INTERNO
  !   ([0, num_threads_j-1]), não um tid global. Com num_threads_k > 1, múltiplos
  !   teams internos teriam threads com tid=0 causando race em ws_pool(0). A
  !   correção adotada é calcular o tid global como:
  !
  !     tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
  !
  !   onde omp_get_ancestor_thread_num(1) retorna o tid do time de nível 1 (outer,
  !   que executa o loop k) e omp_get_thread_num() retorna o tid do time interno.
  !   Com num_threads_k = 1 (produção atual), ancestor(1) == 0 e o tid permanece
  !   backward-compatible com o cálculo antigo. Para multi-ângulo, o índice
  !   percorre [0, num_threads_k*num_threads_j - 1] ⊆ [0, maxthreads-1], seguro.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  allocate(ws_pool(0:maxthreads-1))
  do t = 0, maxthreads-1
    ! Fase 3 — arrays de transmissão/potencial (npt_active × n)
    allocate(ws_pool(t)%Tudw (npt_active, n))
    allocate(ws_pool(t)%Txdw (npt_active, n))
    allocate(ws_pool(t)%Tuup (npt_active, n))
    allocate(ws_pool(t)%Txup (npt_active, n))
    allocate(ws_pool(t)%TEdwz(npt_active, n))
    allocate(ws_pool(t)%TEupz(npt_active, n))
    ! Fase 3b — fatores de onda de commonfactorsMD (npt_active)
    allocate(ws_pool(t)%Mxdw (npt_active))
    allocate(ws_pool(t)%Mxup (npt_active))
    allocate(ws_pool(t)%Eudw (npt_active))
    allocate(ws_pool(t)%Euup (npt_active))
    allocate(ws_pool(t)%FEdwz(npt_active))
    allocate(ws_pool(t)%FEupz(npt_active))
  end do

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Fase 4 — Alocação dos caches de commonarraysMD (shared entre threads)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Fase 4 — alocação com dimensão ntheta: elimina race condition no outer parallel do k.
  ! Custo de memória: ×ntheta (para ntheta=2: ~3,36 MB, era ~1,68 MB — irrelevante).
  allocate(u_cache     (npt_active, n, nf, ntheta))
  allocate(s_cache     (npt_active, n, nf, ntheta))
  allocate(uh_cache    (npt_active, n, nf, ntheta))
  allocate(sh_cache    (npt_active, n, nf, ntheta))
  allocate(RTEdw_cache (npt_active, n, nf, ntheta))
  allocate(RTEup_cache (npt_active, n, nf, ntheta))
  allocate(RTMdw_cache (npt_active, n, nf, ntheta))
  allocate(RTMup_cache (npt_active, n, nf, ntheta))
  allocate(AdmInt_cache(npt_active, n, nf, ntheta))

  ! Fase 4 — eta hoisted para escopo de perfila1DanisoOMP (era recomputado em
  ! cada chamada de fieldsinfreqs antes). Invariante durante todo o modelo.
  allocate(eta_shared(n, 2))
  do ii = 1, n
    eta_shared(ii, 1) = 1.d0 / resist(ii, 1)
    eta_shared(ii, 2) = 1.d0 / resist(ii, 2)
  end do

  ! Exibir configuração de threads apenas no primeiro modelo do lote
  if (modelm == 1) then
    write(*,'(A,I0,A,I0,A,I0,A,I0)') &
      '[OpenMP] maxthreads=', maxthreads, &
      '  threads_angulos(k)=', num_threads_k, &
      '  threads_medidas(j)=', num_threads_j, &
      '  produto=', num_threads_k * num_threads_j

    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! F5 — Validação de frequências arbitrárias
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! Quando use_arb_freq == 0 (padrão), o código aceita qualquer nf mas emite
    ! aviso para nf > 2 como proteção contra uso acidental. Quando habilitado,
    ! valida nf ∈ [1, 16] e exibe as frequências configuradas.
    ! O core já suporta nf arbitrário — caches Phase 4 têm dimensão (npt, n, nf).
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    if (use_arb_freq == 0 .and. nf > 2) then
      write(*,'(A,I0,A)') '[F5 AVISO] nf = ', nf, ' > 2 com use_arbitrary_freq desabilitado.'
      write(*,'(A)')      '           Para nf > 2 sem aviso, defina use_arbitrary_freq = 1 no model.in.'
    end if
    if (use_arb_freq == 1) then
      if (nf < 1 .or. nf > 16) then
        write(*,'(A,I0,A)') '[F5 ERRO] nf = ', nf, ' fora do intervalo válido [1, 16].'
        stop '[F5] nf deve estar entre 1 e 16 com use_arbitrary_freq habilitado'
      end if
      write(*,'(A,I0,A)') '[F5] Frequências arbitrárias habilitadas: nf = ', nf, ' frequência(s)'
      do ii = 1, nf
        write(*,'(A,I0,A,F12.1,A)') '     freq(', ii, ') = ', freq(ii), ' Hz'
      end do
    end if

    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! F7 — Informações sobre antenas inclinadas
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    if (use_tilted == 1 .and. n_tilted > 0) then
      write(*,'(A,I0,A)') '[F7] Antenas inclinadas habilitadas: ', n_tilted, ' configuração(ões)'
      do ii = 1, n_tilted
        write(*,'(A,I0,A,F6.1,A,F6.1,A)') &
          '     tilted(', ii, '): beta=', beta_tilt(ii), '° phi=', phi_tilt(ii), '°'
      end do
      write(*,'(A,I0,A)') '[F7] Saída estendida: 22 + ', 2*n_tilted, ' colunas por registro'
    end if

    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! F6 — Informações sobre compensação midpoint
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    if (use_compensation == 1 .and. n_comp_pairs > 0) then
      write(*,'(A,I0,A)') '[F6] Compensação midpoint habilitada: ', n_comp_pairs, ' par(es)'
      do ii = 1, n_comp_pairs
        write(*,'(A,I0,A,I0,A,I0,A)') &
          '     comp(', ii, '): near=TR', comp_pairs(ii,1), ' far=TR', comp_pairs(ii,2), ''
      end do
      if (nTR < 2) then
        write(*,'(A)') '[F6 AVISO] use_compensation=1 mas nTR < 2. Compensação desabilitada.'
      end if
    end if

    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! Filtro Adaptativo — Informações sobre o filtro selecionado
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    if (filter_type == 1) then
      write(*,'(A,I0,A)') '[FILTRO] Kong (', npt_active, ' pts) — modo rápido para geração de treinamento'
    else if (filter_type == 2) then
      write(*,'(A,I0,A)') '[FILTRO] Anderson (', npt_active, ' pts) — máxima precisão para validação'
    else
      write(*,'(A,I0,A)') '[FILTRO] Werthmuller (', npt_active, ' pts) — padrão (precisão 10⁻⁶)'
    end if
  end if

  allocate(zrho1(ntheta,nmmax,nf,3), cH1(ntheta,nmmax,nf,9))
  allocate(zrho(nf,3), cH(nf,9))
  zrho1 = 0.d0
  cH1 = 0.d0
  allocate(z_rho1(nmmax,nf,3), c_H1(nmmax,nf,9))
  z_rho1 = 0.d0
  c_H1 = 0.d0
  ! F7 — Alocação do array de respostas tilted
  ! Quando habilitado: tamanho completo (ntheta, nmmax, nf, n_tilted) + zerado.
  ! Quando desabilitado: tamanho mínimo (1,1,1,1) SEM zeroing — passado como
  ! assumed-shape a writes_files mas nunca acessado (guard use_tilted==1 interno).
  if (use_tilted == 1 .and. n_tilted > 0) then
    allocate(cH_tilted(ntheta, nmmax, nf, n_tilted))
    cH_tilted = (0.d0, 0.d0)
  else
    allocate(cH_tilted(1, 1, 1, 1))
  end if

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F6 — Alocação dos arrays de compensação midpoint
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! cH_all_tr armazena o tensor completo de TODOS os pares T-R para permitir
  ! o cálculo de compensação após o loop principal. Necessário porque a
  ! compensação requer H_near e H_far simultaneamente — não disponíveis
  ! durante o loop itr que processa um par de cada vez.
  ! F6 — Alocação condicional dos arrays de compensação midpoint.
  ! Quando habilitado: tamanho completo + inicializado.
  ! Quando desabilitado: allocate(0,...) cria descritor alocado com zero bytes
  ! de dados — overhead negligível (~5 descritores sem dados) e silencia
  ! -Wmaybe-uninitialized do gfortran sem impacto na performance.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  if (use_compensation == 1 .and. n_comp_pairs > 0 .and. nTR >= 2) then
    allocate(cH_all_tr(nTR, ntheta, nmmax, nf, 9))
    allocate(zrho_all_tr(nTR, ntheta, nmmax, nf, 3))
    allocate(cH_comp(n_comp_pairs, ntheta, nmmax, nf, 9))
    allocate(phase_diff(n_comp_pairs, ntheta, nmmax, nf, 9))
    allocate(atten_db(n_comp_pairs, ntheta, nmmax, nf, 9))
    cH_all_tr = (0.d0, 0.d0)
    zrho_all_tr = 0.d0
    cH_comp = (0.d0, 0.d0)
    phase_diff = 0.d0
    atten_db = 0.d0
  else
    allocate(cH_all_tr(0, 0, 0, 0, 0))
    allocate(zrho_all_tr(0, 0, 0, 0, 0))
    allocate(cH_comp(0, 0, 0, 0, 0))
    allocate(phase_diff(0, 0, 0, 0, 0))
    allocate(atten_db(0, 0, 0, 0, 0))
  end if

  ! Fase 2 — Hybrid Scheduler: escolha do schedule baseada na característica do loop
  !   • Loop externo `k` (ângulos): carga desigual porque nmed(k) varia com theta(k)
  !     (janela vertical constante mas passo vertical pz = p_med*cos(theta) muda),
  !     portanto `schedule(dynamic)` é apropriado para balanceamento.
  !   • Loop interno `j` (medidas): cada iteração chama commonarraysMD/factors/hmd/vmd
  !     com custo aproximadamente constante (mesmo n, npt, nf). Carga uniforme
  !     favorece `schedule(static)`, que elimina o overhead de sincronização do
  !     dynamic sem penalidade de balanceamento.
  ! Fase 3/PR1-Hygiene: z_rho1 e c_H1 migrados de private → firstprivate (D4).
  ! OpenMP spec 5.x define que cópias private de arrays allocatable têm status
  ! de alocação indefinido. Com firstprivate, cada thread herda a alocação e
  ! os valores do master (inicializados em 0.d0 nas linhas ~158 acima),
  ! garantindo semântica portável. Custo: ~32 KB copiados por thread uma vez
  ! por região paralela — irrelevante para throughput.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Fase 5/5b — Paralelismo adaptativo: single-level (ntheta=1) ou nested (ntheta>1)
  !
  ! Estratégia de particionamento de threads:
  !
  !   ┌────────────────────────────────────────────────────────────────────────┐
  !   │  ntheta = 1 (produção: perfilagem a 0°)                              │
  !   │    → Loop k serial (1 iteração), inner parallel j com maxthreads     │
  !   │    → tid = omp_get_thread_num() direto                               │
  !   │    → Sem overhead de nested fork/join                                │
  !   │                                                                      │
  !   │  ntheta > 1 (multi-ângulo: geosteering)                              │
  !   │    → Outer parallel k com num_threads_k threads (schedule dynamic)   │
  !   │    → Inner parallel j com num_threads_j threads (schedule static)    │
  !   │    → tid = omp_get_ancestor_thread_num(1) * num_threads_j            │
  !   │           + omp_get_thread_num()                                     │
  !   │    → Pré-cômputo commonarraysMD serial dentro de cada k              │
  !   │    → firstprivate(z_rho1, c_H1) para cópias por ângulo              │
  !   └────────────────────────────────────────────────────────────────────────┘
  !
  ! A Fase 3b garante que ws_pool tem 12 campos (6 transmissão + 6 fatores de
  ! onda), eliminando toda pressão de stack mesmo com ntheta > 1 × maxthreads.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Feature 1 — Loop externo sobre pares T-R.
  ! Cada par itr tem seu próprio r_k = dTR(itr) * |sin(θ_k)| e portanto
  ! requer recomputo do cache Fase 4 (commonarraysMD depende de r).
  ! Para nTR=1, este loop executa uma única vez → backward compatible.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  do itr = 1, nTR
  ! Fase 5b: outer parallel com cláusula if(ntheta > 1).
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

    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! Fase 4 — Pré-cômputo de commonarraysMD (serial, uma vez por ângulo k)
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    ! commonarraysMD depende apenas de (r, freq, n, h, eta) — todos invariantes
    ! em j. Por construção r = dTR × |sin(theta_k)| é a distância horizontal
    ! T-R, constante durante a janela de perfilagem (translação rígida da
    ! ferramenta). A sub-rotina commonarraysMD internamente aplica sanitização
    ! if (hordist < eps) hordist = 1.d-2 para o caso theta = 0 (perfilagem
    ! vertical). Aqui passamos r_k direto sem sanitizar — a bit-equivalência
    ! com a versão pré-Fase-4 fica garantida pela sanitização interna.
    !
    ! Redução: nf × nmed = 1.200 chamadas/modelo → nf = 2 chamadas/modelo.
    ! Os caches são lidos (read-only) pelos threads no inner parallel abaixo.
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§��§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    r_k = dTR(itr) * dabs(seno)
    do ii = 1, nf
      omega_i = 2.d0 * pi * freq(ii)
      zeta_i  = cmplx(0.d0, 1.d0, kind=dp) * omega_i * mu
      ! Slice (:,:,ii,k): cada ângulo k escreve em posição independente → thread-safe.
      call commonarraysMD(n, npt_active, r_k, krwJ0J1(:,1), zeta_i, h, eta_shared,   &
                          u_cache(:,:,ii,k),  s_cache(:,:,ii,k),               &
                          uh_cache(:,:,ii,k), sh_cache(:,:,ii,k),              &
                          RTEdw_cache(:,:,ii,k), RTEup_cache(:,:,ii,k),        &
                          RTMdw_cache(:,:,ii,k), RTMup_cache(:,:,ii,k),        &
                          AdmInt_cache(:,:,ii,k))
    end do

    ! Schedule static: iterações têm custo uniforme (mesmos n, npt, nf, nlayers)
    ! Fase 3: cada thread usa ws_pool(tid) para eliminar allocate/deallocate no
    !         hot path (ver magneticdipoles.f08 type :: thread_workspace).
    ! Fase 4: caches compartilhados u_cache,...,AdmInt_cache são lidos por todas
    !         as threads (read-only ⇒ sem race, sem locks).
    ! Débito B3/D7 corrigido: zrho e cH são allocatable com private — OpenMP spec
    ! define status de alocação indefinido para cópias private de allocatables.
    ! Migração para firstprivate garante herança de alocação + valores do master.
    ! Mesmo padrão aplicado a z_rho1/c_H1 em D4 (outer parallel, linha 232).
    ! Fase 5/5b: inner parallel com threads adaptativas.
    ! ntheta=1: maxthreads (single-level), ntheta>1: num_threads_j (nested).
    ! B3/D7: firstprivate(zrho, cH) — allocatable portável.
    ! Fase 2b: schedule(guided, 16) — chunks iniciais grandes (~nmed/threads)
    ! decrescentes até chunk mínimo de 16. Melhora balanceamento em:
    !   - Regimes degradados (poucos threads, nmed grande)
    !   - Multi-ângulo (ntheta>1) com nmed(k) variável entre ângulos
    !   - Custo não-uniforme por iteração (commonfactorsMD varia com camadT)
    ! Chunk=16 preserva localidade de cache L1 (~16 × 19 KB ≈ 300 KB/chunk).
    !$omp parallel do schedule(guided, 16) &
    !$omp&        num_threads(merge(maxthreads, num_threads_j, ntheta == 1)) &
    !$omp&        default(shared) &
    !$omp&        private(j, x, y, z, Tx, Ty, Tz, posTR, tid) &
    !$omp&        firstprivate(zrho, cH)
    do j = 1, nmed(k)
      !----------------------------------------------------
      ! Arranjo TR1:
      !----------------------------------------------------
      ! Quanto o transmissor estiver abaixo dos receptores (configuração dos arranjos de dados da Petrobrás):
      x = 0.d0 + (j-1) * px - Lsen / 2    !considerando-se Tx inicial como sendo 0
      y = 0.d0                             !considerando-se Ty sempre no plano XZ
      z = z1 + (j-1) * pz - Lcos / 2
      Tx = 0.d0 + (j-1) * px + Lsen / 2
      Ty = 0.d0
      Tz = z1 + (j-1) * pz + Lcos / 2
      posTR = (/Tx, Ty, Tz, x, y, z/)
      ! Fase 5/5b: tid adaptativo.
      ! ntheta=1: tid direto (single-level). ntheta>1: tid global (nested, D6).
      if (ntheta == 1) then
        tid = omp_get_thread_num()
      else
        tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
      end if
      ! Fase 4: passa slice (:,:,:,k) — shape (npt_active,n,nf) — compatível com a
      ! interface de fieldsinfreqs_cached_ws (intent(in) dimension(npt,n,nf)).
      ! Thread-safe: cada k lê sua própria fatia do cache, sem conflito.
      call fieldsinfreqs_cached_ws(ws_pool(tid), ang, nf, freq, posTR, dipolo, npt_active, &
                                    krwJ0J1, n, h, prof, resist, eta_shared,        &
                                    u_cache(:,:,:,k),  s_cache(:,:,:,k),            &
                                    uh_cache(:,:,:,k), sh_cache(:,:,:,k),           &
                                    RTEdw_cache(:,:,:,k), RTEup_cache(:,:,:,k),     &
                                    RTMdw_cache(:,:,:,k), RTMup_cache(:,:,:,k),     &
                                    AdmInt_cache(:,:,:,k),                          &
                                    zrho, cH)
      z_rho1(j,:,:) = zrho
      c_H1(j,:,:) = cH
    end do
    !$omp end parallel do
    zrho1(k,1:nmed(k),:,:) = z_rho1
    cH1(k,1:nmed(k),:,:) = c_H1
  end do
  !$omp end parallel do
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F7 — Cálculo das respostas de antenas inclinadas (pós-processamento)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F7 — Cálculo das respostas de antenas inclinadas.
  ! Guard completo: zeroing + cálculo SOMENTE quando F7 habilitado.
  ! Quando desabilitado: zero overhead (sem memset, sem loops).
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  if (use_tilted == 1 .and. n_tilted > 0) then
    ! Zero cH_tilted a cada iteração itr para evitar dados residuais de itr-1.
    cH_tilted = (0.d0, 0.d0)
    do it = 1, n_tilted
      beta_rad = beta_tilt(it) * pi / 18.d1
      phi_rad  = phi_tilt(it) * pi / 18.d1
      do k = 1, ntheta
        do j = 1, nmed(k)
          do i = 1, nf
            cH_tilted(k, j, i, it) = &
              cos(beta_rad) * cH1(k, j, i, 9) + &
              sin(beta_rad) * (cos(phi_rad) * cH1(k, j, i, 3) + &
                               sin(phi_rad) * cH1(k, j, i, 6))
          end do
        end do
      end do
    end do
  end if

  ! Feature 1: escrita de saída por par T-R (dentro do loop itr)
  ! F7: passa cH_tilted para writes_files (condicionalmente alocado)
  call writes_files(modelm, nmaxmodel, mypath, zrho1, cH1, ntheta, theta, nf, freq, nmed, filename, &
                    itr, nTR, use_tilted, n_tilted, beta_tilt, phi_tilt, cH_tilted)

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F6 — Armazenamento do tensor por par T-R para compensação posterior
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Salva cH1 e zrho1 indexados por itr. Após o loop completo, os arrays
  ! cH_all_tr e zrho_all_tr contêm dados de TODOS os pares T-R, necessários
  ! para calcular phase_diff e attenuation entre pares (near, far).
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  if (use_compensation == 1 .and. n_comp_pairs > 0 .and. nTR >= 2) then
    cH_all_tr(itr, :, :, :, :) = cH1
    zrho_all_tr(itr, :, :, :, :) = zrho1
  end if

  end do  ! end do itr = 1, nTR

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! F6 — Cálculo da compensação midpoint (pós-processamento)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Para cada par de compensação (near_itr, far_itr), calcula:
  !   1. Tensor compensado: H_comp = (H_near + H_far) / 2
  !   2. Diferença de fase:  Δφ = arg(H_near) − arg(H_far)  [graus]
  !   3. Atenuação:          Δα = 20·log₁₀(|H_near|/|H_far|)  [dB]
  !
  ! Custo computacional: O(n_comp_pairs × ntheta × nmmax × nf × 9) operações.
  ! Para configuração típica (n_comp_pairs=1, ntheta=1, nmmax=600, nf=2):
  !   ~10.800 operações × ~15 ns ≈ ~162 μs — negligível vs forward model.
  !
  ! Princípio físico (CDR — Compensated Dual Resistivity):
  !   Efeitos ambientais (rugosidade, excentricidade, invasão de lama)
  !   são simétricos em relação ao midpoint T1-T2 e se cancelam na média,
  !   enquanto a resposta da formação (assimétrica) se preserva.
  !   Ref: Schlumberger ARC tool, Baker Hughes OnTrak.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  if (use_compensation == 1 .and. n_comp_pairs > 0 .and. nTR >= 2) then
    do ipair = 1, n_comp_pairs
      i_near = comp_pairs(ipair, 1)
      i_far  = comp_pairs(ipair, 2)
      ! Validação dos índices de pares T-R
      if (i_near < 1 .or. i_near > nTR .or. i_far < 1 .or. i_far > nTR) then
        write(*,'(A,I0,A,I0,A,I0)') &
          '[F6 ERRO] Par de compensação ', ipair, ' com índices inválidos: near=', i_near, ' far=', i_far
        cycle
      end if
      do k = 1, ntheta
        do j = 1, nmed(k)
          do i = 1, nf
            do ic = 1, 9
              ! ── Tensor compensado: média aritmética (CDR clássico) ──
              ! Cancela efeitos de 1ª ordem simétricos ao midpoint.
              cH_comp(ipair, k, j, i, ic) = 0.5d0 * &
                (cH_all_tr(i_near, k, j, i, ic) + cH_all_tr(i_far, k, j, i, ic))

              ! ── Diferença de fase (graus) ──
              ! Δφ = arg(H_near) − arg(H_far), convertido para graus.
              ! atan2 retorna [-π, π]; a diferença é a phase shift entre pares T-R.
              phase_diff(ipair, k, j, i, ic) = &
                (atan2(aimag(cH_all_tr(i_near, k, j, i, ic)), &
                       real(cH_all_tr(i_near, k, j, i, ic))) - &
                 atan2(aimag(cH_all_tr(i_far, k, j, i, ic)), &
                       real(cH_all_tr(i_far, k, j, i, ic)))) * 18.d1 / pi

              ! ── Atenuação (dB) ──
              ! Δα = 20·log₁₀(|H_near|/|H_far|). Protegido contra divisão por zero
              ! com max(|H_far|, 1e-20). Guard 1e-20 é fisicamente motivado:
              ! campos EM em meios condutivos típicos variam entre 1e-6 e 1e-9 A/m²;
              ! valores < 1e-20 são numericamente indistinguíveis de zero no contexto
              ! da simulação. Guard 1e-30 produziria atenuações > 400 dB — não físico.
              abs_near = abs(cH_all_tr(i_near, k, j, i, ic))
              abs_far  = abs(cH_all_tr(i_far, k, j, i, ic))
              atten_db(ipair, k, j, i, ic) = 20.d0 * &
                log10(max(abs_near, 1.d-20) / max(abs_far, 1.d-20))
            end do
          end do
        end do
      end do
    end do

    ! Escrita dos resultados compensados — um arquivo _COMP{ipair}.dat por par
    call writes_compensation_files(modelm, mypath, &
      n_comp_pairs, comp_pairs, ntheta, nf, nmed, filename, &
      cH_comp, phase_diff, atten_db, zrho_all_tr)
  end if

  deallocate(zrho,cH,z_rho1,c_H1)
  if (allocated(cH_tilted)) deallocate(cH_tilted)
  ! F6 — Liberação dos arrays de compensação
  if (allocated(cH_all_tr))    deallocate(cH_all_tr)
  if (allocated(zrho_all_tr))  deallocate(zrho_all_tr)
  if (allocated(cH_comp))      deallocate(cH_comp)
  if (allocated(phase_diff))   deallocate(phase_diff)
  if (allocated(atten_db))     deallocate(atten_db)

  ! Débito B5 corrigido: krwJ0J1 alocado na linha ~99 e nunca desalocado.
  ! Leak de ~9,6 KB/modelo (npt × 3 × 8 bytes = 201 × 3 × 8 ≈ 4,8 KB).
  if (allocated(krwJ0J1)) deallocate(krwJ0J1)

  ! Fase 3 — Liberação do ws_pool após o loop paralelo
  ! Desalocação explícita dos 6 campos por slot + array mestre, mais defensiva
  ! que confiar no deallocate recursivo automático do Fortran 2003.
  do t = 0, maxthreads-1
    ! Fase 3 — campos (npt × n)
    if (allocated(ws_pool(t)%Tudw))  deallocate(ws_pool(t)%Tudw)
    if (allocated(ws_pool(t)%Txdw))  deallocate(ws_pool(t)%Txdw)
    if (allocated(ws_pool(t)%Tuup))  deallocate(ws_pool(t)%Tuup)
    if (allocated(ws_pool(t)%Txup))  deallocate(ws_pool(t)%Txup)
    if (allocated(ws_pool(t)%TEdwz)) deallocate(ws_pool(t)%TEdwz)
    if (allocated(ws_pool(t)%TEupz)) deallocate(ws_pool(t)%TEupz)
    ! Fase 3b — campos (npt)
    if (allocated(ws_pool(t)%Mxdw))  deallocate(ws_pool(t)%Mxdw)
    if (allocated(ws_pool(t)%Mxup))  deallocate(ws_pool(t)%Mxup)
    if (allocated(ws_pool(t)%Eudw))  deallocate(ws_pool(t)%Eudw)
    if (allocated(ws_pool(t)%Euup))  deallocate(ws_pool(t)%Euup)
    if (allocated(ws_pool(t)%FEdwz)) deallocate(ws_pool(t)%FEdwz)
    if (allocated(ws_pool(t)%FEupz)) deallocate(ws_pool(t)%FEupz)
  end do
  deallocate(ws_pool)

  ! Fase 4 — Liberação dos caches de commonarraysMD
  if (allocated(u_cache))      deallocate(u_cache)
  if (allocated(s_cache))      deallocate(s_cache)
  if (allocated(uh_cache))     deallocate(uh_cache)
  if (allocated(sh_cache))     deallocate(sh_cache)
  if (allocated(RTEdw_cache))  deallocate(RTEdw_cache)
  if (allocated(RTEup_cache))  deallocate(RTEup_cache)
  if (allocated(RTMdw_cache))  deallocate(RTMdw_cache)
  if (allocated(RTMup_cache))  deallocate(RTMup_cache)
  if (allocated(AdmInt_cache)) deallocate(AdmInt_cache)
  if (allocated(eta_shared))   deallocate(eta_shared)
  !-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  deallocate(zrho1, cH1)

  ! wtime = omp_get_wtime() - wtime
  ! write(*,'(a,g14.6,1x,a)')' Elapsed wall clock time = ',wtime,'seconds'
  
end subroutine perfila1DanisoOMP
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! subroutine fieldsinfreqs(k, j, ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist, ACx, ACp, zrho, cH)
subroutine fieldsinfreqs(ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist, zrho, cH)
  implicit none
  integer, intent(in) :: nf, npt, n
  character(5), intent(in) :: dipolo
  real(dp), intent(in) :: ang, freqs(nf), krwJ0J1(npt,3), posTR(6), h(n), prof(0:n), resist(n,2)  !, ACx, ACp
  real(dp), dimension(nf,3), intent(out) :: zrho
  complex(dp), dimension(nf,9), intent(out) :: cH

  integer :: i, layerObs, camadT, camadR
  real(dp) :: freq, krJ0J1(npt), wJ0(npt), wJ1(npt)
  real(dp) :: x, y, z, Tx, Ty, Tz, zobs, r, eta(n,2), omega
  complex(dp) :: HxHMD(1,2), HyHMD(1,2), HzHMD(1,2) !só se dipolo = 'hmdxy'. Caso contrário: HxHMD(1,1), HyHMD(1,1), HzHMD(1,1)
  complex(dp) :: zeta, HxVMD, HyVMD, HzVMD, matH(3,3), tH(3,3)
  complex(dp), dimension(npt,1:n) :: u, uh, s, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt
  complex(dp), dimension(npt) :: Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz

  Tx = posTR(1)
  Ty = posTR(2)
  Tz = posTR(3)
  x  = posTR(4)
  y  = posTR(5)
  z  = posTR(6)

  call findlayersTR2well(n, Tz, z, prof(1:n-1), camadT, camadR)
  ! Armazenamento da profundidade do ponto-médio do arranjo T-R1 e suas resistividades verdadeiras
  zobs = (Tz + z) / 2.d0
  layerObs = layer2z_inwell(n, zobs, prof(1:n-1))

  r = sqrt((x - Tx)**2 + (y - Ty)**2)

  krJ0J1 = krwJ0J1(:,1)
  wJ0 = krwJ0J1(:,2)
  wJ1 = krwJ0J1(:,3)

  do i = 1,n
    eta(i,1)  = 1.d0 / resist(i,1)
    eta(i,2)  = 1.d0 / resist(i,2)
  end do
  
  do i = 1, nf
    freq = freqs(i)
    omega = 2.d0 * pi * freq
    zeta = cmplx(0,1.d0,kind=dp) * omega * mu
    zrho(i,:) = (/zobs, resist(layerObs,1), resist(layerObs,2)/)

    ! Cálculo do tensor de indução magnética do arranjo TR triaxial
    !===========================================================================
    call commonarraysMD(n, npt, r, krJ0J1, zeta, h, eta, u, s, uh, sh, &
                      RTEdw, RTEup, RTMdw, RTMup, AdmInt)
    call commonfactorsMD(n, npt, Tz, h, prof, camadT, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, &
                      Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)
    call hmd_TIV_optimized(Tx, Ty, Tz, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, &
                      prof, zeta, eta, x, y, z, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, &
                      Mxdw, Mxup, Eudw, Euup, HxHMD, HyHMD, HzHMD, dipolo)
    call vmd_optimized(Tx, Ty, Tz, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, &
                      prof, zeta, x, y, z, u, uh, AdmInt, RTEdw, RTEup, FEdwz, FEupz, HxVMD, HyVMD, HzVMD)
    !===========================================================================
    matH(1,:) = (/HxHMD(1,1), HyHMD(1,1), HzHMD(1,1)/)
    matH(2,:) = (/HxHMD(1,2), HyHMD(1,2), HzHMD(1,2)/)
    matH(3,:) = (/HxVMD, HyVMD, HzVMD/)
    tH = RtHR(ang, 0.d0, 0.d0, matH)  !tensor de indução magnética da ferramenta triaxial
    ! Armazenamento das partes real e imaginária de Hxx, Hyy e Hzz:
    cH(i,:) = (/ tH(1,1), tH(1,2), tH(1,3), tH(2,1), tH(2,2), tH(2,3), tH(3,1), tH(3,2), tH(3,3) /)
  end do
end subroutine fieldsinfreqs
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Fase 3 — Workspace Pre-allocation (fieldsinfreqs_ws)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
!
! Variante de fieldsinfreqs que recebe um thread_workspace pré-alocado e
! delega as chamadas a hmd_TIV_optimized_ws e vmd_optimized_ws, eliminando
! completamente os allocate/deallocate do hot path paralelo.
!
! Fluxo:
!   ┌──────────────────────────────────────────────────────────────────┐
!   │  perfila1DanisoOMP: !$omp parallel do                            │
!   │    └── tid = omp_get_thread_num()                                │
!   │    └── call fieldsinfreqs_ws(ws_pool(tid), ...)                  │
!   │         ├── commonarraysMD (sem workspace, stack arrays locais)  │
!   │         ├── commonfactorsMD                                      │
!   │         ├── hmd_TIV_optimized_ws(ws, ...)                        │
!   │         └── vmd_optimized_ws(ws, ...)                            │
!   └──────────────────────────────────────────────────────────────────┘
!
! Arrays locais (u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt, Mxdw,
! Mxup, Eudw, Euup, FEdwz, FEupz) permanecem como automatic arrays no stack
! do thread — não contribuem para contenção de malloc e não foram migrados
! para o workspace nesta fase. Ver Fase 3b (opcional, futura) para refatorar
! esses arrays caso modelos com n ≥ 30 camadas causem stack overflow.
!
! Preservação de fieldsinfreqs original: intencional (rollback instantâneo).
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine fieldsinfreqs_ws(ws, ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist, zrho, cH)
  implicit none
  type(thread_workspace), intent(inout) :: ws
  integer, intent(in) :: nf, npt, n
  character(5), intent(in) :: dipolo
  real(dp), intent(in) :: ang, freqs(nf), krwJ0J1(npt,3), posTR(6), h(n), prof(0:n), resist(n,2)
  real(dp), dimension(nf,3), intent(out) :: zrho
  complex(dp), dimension(nf,9), intent(out) :: cH

  integer :: i, layerObs, camadT, camadR
  real(dp) :: freq, krJ0J1(npt), wJ0(npt), wJ1(npt)
  real(dp) :: x, y, z, Tx, Ty, Tz, zobs, r, eta(n,2), omega
  complex(dp) :: HxHMD(1,2), HyHMD(1,2), HzHMD(1,2) !só se dipolo = 'hmdxy'
  complex(dp) :: zeta, HxVMD, HyVMD, HzVMD, matH(3,3), tH(3,3)
  complex(dp), dimension(npt,1:n) :: u, uh, s, sh, RTEdw, RTEup, RTMdw, RTMup, AdmInt
  complex(dp), dimension(npt) :: Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz

  Tx = posTR(1)
  Ty = posTR(2)
  Tz = posTR(3)
  x  = posTR(4)
  y  = posTR(5)
  z  = posTR(6)

  call findlayersTR2well(n, Tz, z, prof(1:n-1), camadT, camadR)
  ! Armazenamento da profundidade do ponto-médio do arranjo T-R1 e suas resistividades verdadeiras
  zobs = (Tz + z) / 2.d0
  layerObs = layer2z_inwell(n, zobs, prof(1:n-1))

  r = sqrt((x - Tx)**2 + (y - Ty)**2)

  krJ0J1 = krwJ0J1(:,1)
  wJ0 = krwJ0J1(:,2)
  wJ1 = krwJ0J1(:,3)

  do i = 1,n
    eta(i,1)  = 1.d0 / resist(i,1)
    eta(i,2)  = 1.d0 / resist(i,2)
  end do

  do i = 1, nf
    freq = freqs(i)
    omega = 2.d0 * pi * freq
    zeta = cmplx(0,1.d0,kind=dp) * omega * mu
    zrho(i,:) = (/zobs, resist(layerObs,1), resist(layerObs,2)/)

    ! Cálculo do tensor de indução magnética do arranjo TR triaxial
    !===========================================================================
    call commonarraysMD(n, npt, r, krJ0J1, zeta, h, eta, u, s, uh, sh, &
                      RTEdw, RTEup, RTMdw, RTMup, AdmInt)
    call commonfactorsMD(n, npt, Tz, h, prof, camadT, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, &
                      Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)
    call hmd_TIV_optimized_ws(ws, Tx, Ty, Tz, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, &
                      prof, zeta, eta, x, y, z, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, &
                      Mxdw, Mxup, Eudw, Euup, HxHMD, HyHMD, HzHMD, dipolo)
    call vmd_optimized_ws(ws, Tx, Ty, Tz, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, &
                      prof, zeta, x, y, z, u, uh, AdmInt, RTEdw, RTEup, FEdwz, FEupz, HxVMD, HyVMD, HzVMD)
    !===========================================================================
    matH(1,:) = (/HxHMD(1,1), HyHMD(1,1), HzHMD(1,1)/)
    matH(2,:) = (/HxHMD(1,2), HyHMD(1,2), HzHMD(1,2)/)
    matH(3,:) = (/HxVMD, HyVMD, HzVMD/)
    tH = RtHR(ang, 0.d0, 0.d0, matH)  !tensor de indução magnética da ferramenta triaxial
    ! Armazenamento das partes real e imaginária de Hxx, Hyy e Hzz:
    cH(i,:) = (/ tH(1,1), tH(1,2), tH(1,3), tH(2,1), tH(2,2), tH(2,3), tH(3,1), tH(3,2), tH(3,3) /)
  end do
end subroutine fieldsinfreqs_ws
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Fase 4 — fieldsinfreqs_cached_ws: usa caches pré-computados de commonarraysMD
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
!
! Variante de fieldsinfreqs_ws que, em vez de chamar commonarraysMD a cada
! invocação, recebe os 9 arrays de cache pré-computados (um por frequência)
! como argumentos intent(in). As slices u_c(:,:,i), s_c(:,:,i), ... são
! passadas para as rotinas hmd_TIV_optimized_ws e vmd_optimized_ws sem cópia
! (contíguas em column-major Fortran).
!
! Redução efetiva: 1.200 chamadas/modelo de commonarraysMD → nf chamadas
! (executadas fora de fieldsinfreqs_cached_ws, no pré-cômputo serial do loop k).
!
! commonfactorsMD PERMANECE inline (hot path) porque depende de camadT e Tz,
! ambos variáveis em j. Será tratada em Fase 6 (cache por camadT).
!
! Ref: docs/reference/analise_paralelismo_cpu_fortran.md §7 Fase 4
!      docs/reference/relatorio_fase4_fortran.md
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine fieldsinfreqs_cached_ws(ws, ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, &
                                    n, h, prof, resist, eta_in,                      &
                                    u_c, s_c, uh_c, sh_c,                            &
                                    RTEdw_c, RTEup_c, RTMdw_c, RTMup_c, AdmInt_c,    &
                                    zrho, cH)
  ! INPUT:
  !   ws        : workspace pré-alocado por thread (Fase 3)
  !   ang, nf, freqs, posTR, dipolo, npt, krwJ0J1, n, h, prof, resist:
  !               idênticos a fieldsinfreqs_ws
  !   eta_in(n,2): admitividades hoisted (1/resist) — evita recomputo por chamada
  !   u_c..AdmInt_c(npt,n,nf): arrays pré-computados por commonarraysMD no
  !               escopo do caller (perfila1DanisoOMP), um slot por frequência
  ! OUTPUT:
  !   zrho(nf,3), cH(nf,9): idênticos a fieldsinfreqs_ws
  implicit none
  type(thread_workspace), intent(inout) :: ws
  integer, intent(in) :: nf, npt, n
  character(5), intent(in) :: dipolo
  real(dp), intent(in) :: ang, freqs(nf), krwJ0J1(npt,3), posTR(6), h(n), prof(0:n)
  real(dp), intent(in) :: resist(n,2), eta_in(n,2)
  complex(dp), dimension(npt,n,nf), intent(in) :: u_c, s_c, uh_c, sh_c
  complex(dp), dimension(npt,n,nf), intent(in) :: RTEdw_c, RTEup_c, RTMdw_c, RTMup_c
  complex(dp), dimension(npt,n,nf), intent(in) :: AdmInt_c
  real(dp), dimension(nf,3), intent(out) :: zrho
  complex(dp), dimension(nf,9), intent(out) :: cH

  integer :: i, layerObs, camadT, camadR
  ! Débito B1 corrigido: eliminada cópia redundante krJ0J1/wJ0/wJ1 = krwJ0J1(:,1..3).
  ! As slices krwJ0J1(:,1), krwJ0J1(:,2), krwJ0J1(:,3) são passadas diretamente
  ! para hmd_TIV_optimized_ws e vmd_optimized_ws abaixo. Em column-major, a
  ! primeira dimensão (npt) é contígua — cada slice (:,k) é contígua em memória,
  ! sem necessidade de cópia temporária pelo compilador.
  real(dp) :: freq
  real(dp) :: x, y, z, Tx, Ty, Tz, zobs, omega
  complex(dp) :: HxHMD(1,2), HyHMD(1,2), HzHMD(1,2) !só se dipolo = 'hmdxy'
  complex(dp) :: zeta, HxVMD, HyVMD, HzVMD, matH(3,3), tH(3,3)
  ! Fase 3b: Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz movidos de automatic (stack)
  ! para ws%Mxdw etc. (heap, via thread_workspace). Eliminação de ~19 KB/thread
  ! de pressão de stack. Robustez para n ≥ 30 camadas + muitos threads.

  Tx = posTR(1)
  Ty = posTR(2)
  Tz = posTR(3)
  x  = posTR(4)
  y  = posTR(5)
  z  = posTR(6)

  call findlayersTR2well(n, Tz, z, prof(1:n-1), camadT, camadR)
  ! Armazenamento da profundidade do ponto-médio do arranjo T-R1 e suas resistividades verdadeiras
  zobs = (Tz + z) / 2.d0
  layerObs = layer2z_inwell(n, zobs, prof(1:n-1))

  do i = 1, nf
    freq  = freqs(i)
    omega = 2.d0 * pi * freq
    zeta  = cmplx(0,1.d0,kind=dp) * omega * mu
    zrho(i,:) = (/zobs, resist(layerObs,1), resist(layerObs,2)/)

    ! Cálculo do tensor de indução magnética do arranjo TR triaxial
    !===========================================================================
    ! commonarraysMD ELIMINADO — slices dos caches passadas diretamente.
    ! commonfactorsMD permanece (depende de camadT, variável em j).
    call commonfactorsMD(n, npt, Tz, h, prof, camadT, &
                         u_c(:,:,i), s_c(:,:,i), uh_c(:,:,i), sh_c(:,:,i), &
                         RTEdw_c(:,:,i), RTEup_c(:,:,i),                    &
                         RTMdw_c(:,:,i), RTMup_c(:,:,i),                    &
                         ws%Mxdw, ws%Mxup, ws%Eudw, ws%Euup, ws%FEdwz, ws%FEupz)
    call hmd_TIV_optimized_ws(ws, Tx, Ty, Tz, n, camadR, camadT, npt,       &
                      krwJ0J1(:,1), krwJ0J1(:,2), krwJ0J1(:,3), h,        &
                      prof, zeta, eta_in, x, y, z,                         &
                      u_c(:,:,i), s_c(:,:,i), uh_c(:,:,i), sh_c(:,:,i),   &
                      RTEdw_c(:,:,i), RTEup_c(:,:,i),                      &
                      RTMdw_c(:,:,i), RTMup_c(:,:,i),                      &
                      ws%Mxdw, ws%Mxup, ws%Eudw, ws%Euup, HxHMD, HyHMD, HzHMD, dipolo)
    call vmd_optimized_ws(ws, Tx, Ty, Tz, n, camadR, camadT, npt,         &
                      krwJ0J1(:,1), krwJ0J1(:,2), krwJ0J1(:,3), h,        &
                      prof, zeta, x, y, z,                                 &
                      u_c(:,:,i), uh_c(:,:,i), AdmInt_c(:,:,i),            &
                      RTEdw_c(:,:,i), RTEup_c(:,:,i),                      &
                      ws%FEdwz, ws%FEupz, HxVMD, HyVMD, HzVMD)
    !===========================================================================
    matH(1,:) = (/HxHMD(1,1), HyHMD(1,1), HzHMD(1,1)/)
    matH(2,:) = (/HxHMD(1,2), HyHMD(1,2), HzHMD(1,2)/)
    matH(3,:) = (/HxVMD, HyVMD, HzVMD/)
    tH = RtHR(ang, 0.d0, 0.d0, matH)  !tensor de indução magnética da ferramenta triaxial
    ! Armazenamento das partes real e imaginária de Hxx, Hyy e Hzz:
    cH(i,:) = (/ tH(1,1), tH(1,2), tH(1,3), tH(2,1), tH(2,2), tH(2,3), tH(3,1), tH(3,2), tH(3,3) /)
  end do
end subroutine fieldsinfreqs_cached_ws
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine writes_files(modelm, nmaxmodel, mypath, zrho, cH, nt, theta, nf, freq, nmeds, filename, &
                        itr, nTR, use_tilted, n_tilted_in, beta_tilt_in, phi_tilt_in, cH_tilted_in)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Versão 8.0: escrita de saída com suporte a Feature 1 (Multi-TR) e F7 (Tilted).
  !   - Feature 1: sufixo _TR{itr} para nTR > 1, sem sufixo para nTR == 1
  !   - F7: quando use_tilted == 1, anexa Re/Im de H_tilted para cada configuração
  !         ao registro binário, resultando em 22 + 2×n_tilted colunas.
  !         O arquivo .out inclui metadados das antenas inclinadas.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  implicit none
  character(*), intent(in) :: mypath
  integer, intent(in) :: modelm, nmaxmodel, nt, nf, itr, nTR
  integer, intent(in) :: nmeds(nt)
  real(dp), intent(in) :: theta(nt), freq(nf)
  real(dp), dimension(:,:,:,:), intent(in) :: zrho
  complex(dp), dimension(:,:,:,:), intent(in) :: cH
  character(*), intent(in) :: filename
  ! F7 — Parâmetros de antenas inclinadas
  integer, intent(in) :: use_tilted, n_tilted_in
  real(dp), intent(in) :: beta_tilt_in(:), phi_tilt_in(:)
  complex(dp), dimension(:,:,:,:), intent(in) :: cH_tilted_in  ! (nt, nmmax, nf, n_tilted)

  integer :: k, j, i, exec, it
  character(len=:), allocatable :: infomodels, fileTR
  character(len=10) :: tr_suffix
  logical :: file_exists

  if (modelm == nmaxmodel) then
    infomodels = mypath//trim('info')//trim(adjustl(filename))//'.out'
    open(unit = 10, file = infomodels, status = 'replace', action = 'write')
    write(10,*) nt, nf, nmaxmodel
    write(10,*) (/(theta(i),i=1,nt)/)
    write(10,*) (/(freq(i),i=1,nf)/)
    write(10,*) (/(nmeds(i),i=1,nt)/)
    ! F7 — Metadados de antenas inclinadas no arquivo .out
    ! Linha 5: use_tilted, n_tilted (0 0 quando desabilitado)
    write(10,*) use_tilted, n_tilted_in
    if (use_tilted == 1 .and. n_tilted_in > 0) then
      write(10,*) (/(beta_tilt_in(it), it=1,n_tilted_in)/)
      write(10,*) (/(phi_tilt_in(it), it=1,n_tilted_in)/)
    end if
    close(10)
  end if
  ! Arquivos: sufixo _TR{itr} para nTR > 1, sem sufixo para nTR == 1
  if (nTR > 1) then
    write(tr_suffix, '(A,I0)') '_TR', itr
    fileTR = mypath//trim(adjustl(filename))//trim(tr_suffix)//'.dat'
  else
    fileTR = mypath//trim(adjustl(filename))//'.dat'
  end if

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Correção Débito 1 — Abertura Condicional do Arquivo de Saída
  ! Ref: docs/reference/relatorio_fase0_fase1_fortran.md §4.1
  !
  ! Bug original: o código usava `status='unknown', position='append'` em todas
  ! as chamadas, incluindo modelm=1. Se o arquivo de saída já existisse de uma
  ! execução anterior do loop externo (fifthBuildTIVModels.py), os novos dados
  ! eram CONCATENADOS aos antigos, produzindo arquivos com K × N registros em
  ! vez de N. Isto invalidava benchmarks e treinamento silenciosamente.
  !
  ! Correção: abertura com detecção de existência do arquivo + flag modelm.
  ! Lógica de decisão:
  !   ┌─────────────────────┬─────────────────────┬──────────────────────────┐
  !   │ modelm              │ arquivo existe?     │ ação                      │
  !   ├─────────────────────┼─────────────────────┼──────────────────────────┤
  !   │ == 1 (1º do lote)   │ não importa         │ status='replace' (wipe)   │
  !   │ > 1 (subsequente)   │ sim                 │ status='old' + append     │
  !   │ > 1 (subsequente)   │ não                 │ status='replace' (safe)   │
  !   └─────────────────────┴─────────────────────┴──────────────────────────┘
  !
  ! Casos de uso cobertos:
  !   1. Loop de produção (fifthBuildTIVModels.py): chama com modelm=1,2,...,N.
  !      Primeiro cria arquivo novo; demais anexam. Rerun sobrescreve corretamente.
  !   2. Benchmark ou execução isolada (model.in com modelm=nmaxmodel=1000):
  !      modelm=1000, arquivo não existe → fallback replace (benchmark isolado).
  !   3. Defensivo: modelm>1 com arquivo ausente (ex: interrupção do lote)
  !      → recria arquivo em vez de abortar com erro 'old, file does not exist'.
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  inquire(file = fileTR, exist = file_exists)
  if (modelm == 1 .or. .not. file_exists) then
    ! Primeiro modelo do lote OU arquivo ausente: cria novo (sobrescreve se existir)
    open(unit = 1000, iostat = exec, file = fileTR, form = 'unformatted', &
         access = 'stream', status = 'replace', action = 'write')
  else
    ! Modelos subsequentes com arquivo existente: anexa
    open(unit = 1000, iostat = exec, file = fileTR, form = 'unformatted', &
         access = 'stream', status = 'old', position = 'append', action = 'write')
  end if

  do k = 1, nt
    do j = 1, nf
      do i = 1, nmeds(k)
        ! Registro padrão: 1 int32 + 21 float64 = 172 bytes (22 colunas)
        write(1000) i, zrho(k,i,j,1), zrho(k,i,j,2), zrho(k,i,j,3), &
                   real(cH(k,i,j,1)), aimag(cH(k,i,j,1)), real(cH(k,i,j,2)), aimag(cH(k,i,j,2)), &
                   real(cH(k,i,j,3)), aimag(cH(k,i,j,3)), real(cH(k,i,j,4)), aimag(cH(k,i,j,4)), &
                   real(cH(k,i,j,5)), aimag(cH(k,i,j,5)), real(cH(k,i,j,6)), aimag(cH(k,i,j,6)), &
                   real(cH(k,i,j,7)), aimag(cH(k,i,j,7)), real(cH(k,i,j,8)), aimag(cH(k,i,j,8)), &
                   real(cH(k,i,j,9)), aimag(cH(k,i,j,9))
        ! F7 — Extensão tilted: 2×n_tilted float64 adicionais por registro
        ! Formato binário stream: dados contíguos, sem delimitadores de registro.
        ! Cada configuração tilted adiciona Re(H_tilted) + Im(H_tilted) = 16 bytes.
        ! Total por registro: 172 + n_tilted × 16 bytes.
        if (use_tilted == 1 .and. n_tilted_in > 0) then
          do it = 1, n_tilted_in
            write(1000) real(cH_tilted_in(k,i,j,it)), aimag(cH_tilted_in(k,i,j,it))
          end do
        end if
      end do
    end do
  end do
  close(unit = 1000)
  
end subroutine writes_files
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! F6 — Sub-rotina de escrita dos resultados de compensação midpoint
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine writes_compensation_files(modelm, mypath, &
    n_comp_pairs, comp_pairs, nt, nf, nmeds, filename, &
    cH_comp, phase_diff, atten_db, zrho_all_tr)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Escreve os resultados da compensação midpoint (F6) em arquivos binários.
  !
  ! Para cada par de compensação (near_itr, far_itr), gera:
  !   {filename}_COMP{ipair}.dat — arquivo binário stream
  !
  ! Formato do registro (29 colunas):
  !   col 0:  i (int32) — índice da medida
  !   col 1:  z_obs (float64) — profundidade do ponto médio
  !   col 2:  rho_h_near (float64) — resistividade horizontal (par near)
  !   col 3:  rho_v_near (float64) — resistividade vertical (par near)
  !   col 4-21:  Re/Im(H_comp(1:9)) — 9 componentes do tensor compensado
  !   col 22-30: phase_diff(1:9) — diferença de fase por componente (graus)
  !   (total: 1 int32 + 28 float64 = 228 bytes/registro)
  !
  ! Nota: attenuation (dB) é armazenada num arquivo separado _COMP{ipair}_ATT.dat
  ! para manter compatibilidade de formato com leitores existentes.
  !
  ! Abertura condicional: mesma lógica de writes_files (Débito 1).
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  implicit none
  character(*), intent(in) :: mypath
  integer, intent(in) :: modelm, nt, nf, n_comp_pairs
  integer, intent(in) :: nmeds(nt), comp_pairs(:,:)
  character(*), intent(in) :: filename
  complex(dp), dimension(:,:,:,:,:), intent(in) :: cH_comp      ! (n_comp_pairs, nt, nmmax, nf, 9)
  real(dp), dimension(:,:,:,:,:), intent(in)    :: phase_diff    ! (n_comp_pairs, nt, nmmax, nf, 9)
  real(dp), dimension(:,:,:,:,:), intent(in)    :: atten_db      ! (n_comp_pairs, nt, nmmax, nf, 9)
  real(dp), dimension(:,:,:,:,:), intent(in)    :: zrho_all_tr   ! (nTR, nt, nmmax, nf, 3)

  integer :: ipair, k, j, i, ic, exec, i_near
  character(len=:), allocatable :: fileComp, fileAtt
  character(len=20) :: comp_suffix
  logical :: file_exists

  do ipair = 1, n_comp_pairs
    i_near = comp_pairs(ipair, 1)

    ! ── Arquivo do tensor compensado + phase_diff ──
    write(comp_suffix, '(A,I0)') '_COMP', ipair
    fileComp = mypath//trim(adjustl(filename))//trim(comp_suffix)//'.dat'

    inquire(file = fileComp, exist = file_exists)
    if (modelm == 1 .or. .not. file_exists) then
      open(unit = 1100 + ipair, iostat = exec, file = fileComp, form = 'unformatted', &
           access = 'stream', status = 'replace', action = 'write')
    else
      open(unit = 1100 + ipair, iostat = exec, file = fileComp, form = 'unformatted', &
           access = 'stream', status = 'old', position = 'append', action = 'write')
    end if

    do k = 1, nt
      do j = 1, nf
        do i = 1, nmeds(k)
          ! Registro: 1 int32 + 3 float64 (zobs, rho_h, rho_v) +
          !           18 float64 (Re/Im H_comp 9 comp) +
          !           9 float64 (phase_diff 9 comp)
          write(1100 + ipair) i, &
            zrho_all_tr(i_near, k, i, j, 1), &
            zrho_all_tr(i_near, k, i, j, 2), &
            zrho_all_tr(i_near, k, i, j, 3), &
            (real(cH_comp(ipair, k, i, j, ic)), aimag(cH_comp(ipair, k, i, j, ic)), ic=1,9), &
            (phase_diff(ipair, k, i, j, ic), ic=1,9)
        end do
      end do
    end do
    close(unit = 1100 + ipair)

    ! ── Arquivo de atenuação (dB) ──
    fileAtt = mypath//trim(adjustl(filename))//trim(comp_suffix)//'_ATT.dat'

    inquire(file = fileAtt, exist = file_exists)
    if (modelm == 1 .or. .not. file_exists) then
      open(unit = 1200 + ipair, iostat = exec, file = fileAtt, form = 'unformatted', &
           access = 'stream', status = 'replace', action = 'write')
    else
      open(unit = 1200 + ipair, iostat = exec, file = fileAtt, form = 'unformatted', &
           access = 'stream', status = 'old', position = 'append', action = 'write')
    end if

    do k = 1, nt
      do j = 1, nf
        do i = 1, nmeds(k)
          ! Registro: 1 int32 + 3 float64 (zobs, rho_h, rho_v) +
          !           9 float64 (atten_db por componente)
          write(1200 + ipair) i, &
            zrho_all_tr(i_near, k, i, j, 1), &
            zrho_all_tr(i_near, k, i, j, 2), &
            zrho_all_tr(i_near, k, i, j, 3), &
            (atten_db(ipair, k, i, j, ic), ic=1,9)
        end do
      end do
    end do
    close(unit = 1200 + ipair)
  end do

end subroutine writes_compensation_files
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

subroutine write_results(results, nk, nj, ni, arq, filename)
  implicit none
  integer, dimension(:,:,:), intent(in) :: results
  integer, intent(in) :: nk, nj, ni, arq
  character(len=*), intent(in) :: filename
  integer :: k, j, i
  ! integer :: arq

  open(unit = arq, file = filename, status = "replace", action = "write")

  do k = 1, nk
    do j = 1, nj
      do i = 1, ni
        ! if (results(k, j, i) /= 0) then
          write(arq, '(i1,1x,i1,1x,i1,1x,i2)') k, j, i, results(k, j, i)
        ! end if
      end do
    end do
  end do

  close(arq)
end subroutine write_results
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
end module DManisoTIV
