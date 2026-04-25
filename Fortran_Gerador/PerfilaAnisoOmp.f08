module DManisoTIV
use parameters
use filterscommonbase
use utils
use magneticdipoles
use omp_lib
contains
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, dTR, p_med, &
                             n, resist, esp, filename)
  implicit none
  character(*), intent(in) :: mypath
  integer, intent(in) ::  modelm, nmaxmodel, nf, ntheta, n
  real(dp), intent(in) :: freq(nf), theta(ntheta), h1, tj, dTR, p_med, resist(n,2), esp(n) !, hn
  character(*), intent(in) :: filename

  integer :: i, j, k, nmmax
  real(dp) :: thetamin, thetaplu, thetarad
  ! real(dp) :: tj
  character(5) :: dipolo
  integer, parameter :: npt = 201
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
  integer :: t, tid

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
  complex(dp), allocatable :: u_cache(:,:,:), s_cache(:,:,:)
  complex(dp), allocatable :: uh_cache(:,:,:), sh_cache(:,:,:)
  complex(dp), allocatable :: RTEdw_cache(:,:,:), RTEup_cache(:,:,:)
  complex(dp), allocatable :: RTMdw_cache(:,:,:), RTMup_cache(:,:,:)
  complex(dp), allocatable :: AdmInt_cache(:,:,:)
  real(dp)    :: r_k, omega_i
  complex(dp) :: zeta_i
  real(dp), allocatable :: eta_shared(:,:)  ! (n, 2) — hoisted de fieldsinfreqs (B2)
  integer     :: ii
  ! real(dp) :: wtime

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

  call J0J1Wer(npt, krJ0J1, wJ0, wJ1)
  allocate(krwJ0J1(npt,3))
  do i = 1, npt
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
    allocate(ws_pool(t)%Tudw (npt, n))
    allocate(ws_pool(t)%Txdw (npt, n))
    allocate(ws_pool(t)%Tuup (npt, n))
    allocate(ws_pool(t)%Txup (npt, n))
    allocate(ws_pool(t)%TEdwz(npt, n))
    allocate(ws_pool(t)%TEupz(npt, n))
  end do

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! Fase 4 — Alocação dos caches de commonarraysMD (shared entre threads)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  allocate(u_cache     (npt, n, nf))
  allocate(s_cache     (npt, n, nf))
  allocate(uh_cache    (npt, n, nf))
  allocate(sh_cache    (npt, n, nf))
  allocate(RTEdw_cache (npt, n, nf))
  allocate(RTEup_cache (npt, n, nf))
  allocate(RTMdw_cache (npt, n, nf))
  allocate(RTMup_cache (npt, n, nf))
  allocate(AdmInt_cache(npt, n, nf))

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
  end if

  allocate(zrho1(ntheta,nmmax,nf,3), cH1(ntheta,nmmax,nf,9))
  allocate(zrho(nf,3), cH(nf,9))
  zrho1 = 0.d0
  cH1 = 0.d0
  allocate(z_rho1(nmmax,nf,3), c_H1(nmmax,nf,9))
  z_rho1 = 0.d0
  c_H1 = 0.d0
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
  !$omp parallel do schedule(dynamic) num_threads(num_threads_k) &
  !$omp&        private(k,ang,seno,coss,px,pz,Lsen,Lcos) &
  !$omp&        firstprivate(z_rho1,c_H1)
  do k = 1, ntheta
    ang = theta(k) * pi / 18.d1
    seno = sin(ang)
    coss = cos(ang)
    px = p_med * seno
    pz = p_med * coss
    Lsen = dTR * seno
    Lcos = dTR * coss

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
    !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
    r_k = dTR * dabs(seno)
    do ii = 1, nf
      omega_i = 2.d0 * pi * freq(ii)
      zeta_i  = cmplx(0.d0, 1.d0, kind=dp) * omega_i * mu
      call commonarraysMD(n, npt, r_k, krwJ0J1(:,1), zeta_i, h, eta_shared,   &
                          u_cache(:,:,ii),  s_cache(:,:,ii),                   &
                          uh_cache(:,:,ii), sh_cache(:,:,ii),                  &
                          RTEdw_cache(:,:,ii), RTEup_cache(:,:,ii),            &
                          RTMdw_cache(:,:,ii), RTMup_cache(:,:,ii),            &
                          AdmInt_cache(:,:,ii))
    end do

    ! Schedule static: iterações têm custo uniforme (mesmos n, npt, nf, nlayers)
    ! Fase 3: cada thread usa ws_pool(tid) para eliminar allocate/deallocate no
    !         hot path (ver magneticdipoles.f08 type :: thread_workspace).
    ! Fase 4: caches compartilhados u_cache,...,AdmInt_cache são lidos por todas
    !         as threads (read-only ⇒ sem race, sem locks).
    !$omp parallel do schedule(static) num_threads(num_threads_j) &
    !$omp&        default(shared) &
    !$omp&        private(j, x, y, z, Tx, Ty, Tz, posTR, zrho, cH, tid)
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
      ! Fase 3 + D6 corrigido (PR1-Hygiene): tid GLOBAL do workspace pool.
      ! Com num_threads_k==1 (ntheta=1 produção), ancestor(1)==0 ⇒ tid == inner_tid.
      ! Com num_threads_k>1 (multi-ângulo futuro), tid percorre [0, maxthreads-1].
      tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()
      ! Fase 4: chamada usa os caches pré-computados (shared) em vez de
      ! recalcular commonarraysMD a cada iteração j.
      call fieldsinfreqs_cached_ws(ws_pool(tid), ang, nf, freq, posTR, dipolo, npt, &
                                    krwJ0J1, n, h, prof, resist, eta_shared,        &
                                    u_cache, s_cache, uh_cache, sh_cache,           &
                                    RTEdw_cache, RTEup_cache,                        &
                                    RTMdw_cache, RTMup_cache, AdmInt_cache,          &
                                    zrho, cH)
      z_rho1(j,:,:) = zrho
      c_H1(j,:,:) = cH
    end do
    !$omp end parallel do
    zrho1(k,1:nmed(k),:,:) = z_rho1
    cH1(k,1:nmed(k),:,:) = c_H1
  end do
  !$omp end parallel do
  ! Fase 3/PR1-Hygiene: removido !$omp barrier órfão que estava fora de região
  ! paralela (débito D5 da Fase 3). O !$omp end parallel do acima já contém
  ! uma barreira implícita — a diretiva explícita era redundante e semanticamente
  ! inválida fora de uma região parallel ativa.
  deallocate(zrho,cH,z_rho1,c_H1)

  ! Fase 3 — Liberação do ws_pool após o loop paralelo
  ! Desalocação explícita dos 6 campos por slot + array mestre, mais defensiva
  ! que confiar no deallocate recursivo automático do Fortran 2003.
  do t = 0, maxthreads-1
    if (allocated(ws_pool(t)%Tudw))  deallocate(ws_pool(t)%Tudw)
    if (allocated(ws_pool(t)%Txdw))  deallocate(ws_pool(t)%Txdw)
    if (allocated(ws_pool(t)%Tuup))  deallocate(ws_pool(t)%Tuup)
    if (allocated(ws_pool(t)%Txup))  deallocate(ws_pool(t)%Txup)
    if (allocated(ws_pool(t)%TEdwz)) deallocate(ws_pool(t)%TEdwz)
    if (allocated(ws_pool(t)%TEupz)) deallocate(ws_pool(t)%TEupz)
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
  ! write(*,*)'Passando para a escrita dos arquivos'
  call writes_files(modelm, nmaxmodel, mypath, zrho1, cH1, ntheta, theta, nf, freq, nmed, filename)
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
  real(dp) :: freq, krJ0J1(npt), wJ0(npt), wJ1(npt)
  real(dp) :: x, y, z, Tx, Ty, Tz, zobs, omega
  complex(dp) :: HxHMD(1,2), HyHMD(1,2), HzHMD(1,2) !só se dipolo = 'hmdxy'
  complex(dp) :: zeta, HxVMD, HyVMD, HzVMD, matH(3,3), tH(3,3)
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

  ! r local nem é usado aqui — estaria redundante, pois commonarraysMD já foi
  ! chamada pelo caller (perfila1DanisoOMP) com o r correto.

  krJ0J1 = krwJ0J1(:,1)
  wJ0    = krwJ0J1(:,2)
  wJ1    = krwJ0J1(:,3)

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
                         Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz)
    call hmd_TIV_optimized_ws(ws, Tx, Ty, Tz, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, &
                      prof, zeta, eta_in, x, y, z,                                  &
                      u_c(:,:,i), s_c(:,:,i), uh_c(:,:,i), sh_c(:,:,i),             &
                      RTEdw_c(:,:,i), RTEup_c(:,:,i),                                &
                      RTMdw_c(:,:,i), RTMup_c(:,:,i),                                &
                      Mxdw, Mxup, Eudw, Euup, HxHMD, HyHMD, HzHMD, dipolo)
    call vmd_optimized_ws(ws, Tx, Ty, Tz, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, &
                      prof, zeta, x, y, z,                                          &
                      u_c(:,:,i), uh_c(:,:,i), AdmInt_c(:,:,i),                     &
                      RTEdw_c(:,:,i), RTEup_c(:,:,i),                                &
                      FEdwz, FEupz, HxVMD, HyVMD, HzVMD)
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
subroutine writes_files(modelm, nmaxmodel, mypath, zrho, cH, nt, theta, nf, freq, nmeds, filename)
  implicit none
  character(*), intent(in) :: mypath
  integer, intent(in) :: modelm, nmaxmodel, nt, nf
  integer, intent(in) :: nmeds(nt)
  real(dp), intent(in) :: theta(nt), freq(nf)
  real(dp), dimension(:,:,:,:), intent(in) :: zrho
  complex(dp), dimension(:,:,:,:), intent(in) :: cH
  character(*), intent(in) :: filename
 
  integer :: k, j, i, exec
  character(len=:), allocatable :: infomodels, fileTR
  logical :: file_exists

  if (modelm == nmaxmodel) then
    infomodels = mypath//trim('info')//trim(adjustl(filename))//'.out'
    open(unit = 10, file = infomodels, status = 'replace', action = 'write')
    write(10,*)nt,nf,nmaxmodel
    write(10,*)(/(theta(i),i=1,nt)/)
    write(10,*)(/(freq(i),i=1,nf)/)
    write(10,*)(/(nmeds(i),i=1,nt)/)
    close(10)
  end if
  ! Arquivos:
  fileTR = mypath//trim(adjustl(filename))//'.dat'

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
        write(1000)i, zrho(k,i,j,1), zrho(k,i,j,2), zrho(k,i,j,3), &
                   real(cH(k,i,j,1)), aimag(cH(k,i,j,1)), real(cH(k,i,j,2)), aimag(cH(k,i,j,2)), &
                   real(cH(k,i,j,3)), aimag(cH(k,i,j,3)), real(cH(k,i,j,4)), aimag(cH(k,i,j,4)), &
                   real(cH(k,i,j,5)), aimag(cH(k,i,j,5)), real(cH(k,i,j,6)), aimag(cH(k,i,j,6)), &
                   real(cH(k,i,j,7)), aimag(cH(k,i,j,7)), real(cH(k,i,j,8)), aimag(cH(k,i,j,8)), &
                   real(cH(k,i,j,9)), aimag(cH(k,i,j,9))
        ! write(1000)i, freq(j), theta(k), zrho(k,i,j,1), zrho(k,i,j,2), zrho(k,i,j,3), &
        !            real(cH(k,i,j,1)), aimag(cH(k,i,j,1)), real(cH(k,i,j,2)), aimag(cH(k,i,j,2)), &
        !            real(cH(k,i,j,3)), aimag(cH(k,i,j,3)), real(cH(k,i,j,4)), aimag(cH(k,i,j,4)), &
        !            real(cH(k,i,j,5)), aimag(cH(k,i,j,5)), real(cH(k,i,j,6)), aimag(cH(k,i,j,6)), &
        !            real(cH(k,i,j,7)), aimag(cH(k,i,j,7)), real(cH(k,i,j,8)), aimag(cH(k,i,j,8)), &
        !            real(cH(k,i,j,9)), aimag(cH(k,i,j,9))
      end do
    end do
  end do
  close(unit = 1000)
  
end subroutine writes_files
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
