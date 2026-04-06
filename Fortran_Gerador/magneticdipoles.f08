module magneticdipoles
  use parameters
  implicit none

  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  ! type :: thread_workspace — Workspace pré-alocado por thread (Fase 3)
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  !
  ! Agrega os seis arrays que, nas rotinas originais (hmd_TIV_optimized e
  ! vmd_optimized), eram alocados/desalocados dinamicamente a cada chamada
  ! no hot path paralelo. Cada iteração do loop interno (medidas j) chamava
  ! as duas rotinas, resultando em ~7.200 a ~12.000 chamadas de allocate/
  ! deallocate por modelo (nf × nmed × 6 arrays). Em 8 threads paralelas,
  ! essas chamadas contendem o mutex do heap (malloc lock), limitando a
  ! escalabilidade observada nas Fases 0–2.
  !
  ! Layout e uso:
  !
  !   ┌────────────────────────────────────────────────────────────────────┐
  !   │  perfila1DanisoOMP (serial, antes do loop paralelo):              │
  !   │    ① allocate(ws_pool(0:maxthreads-1))                             │
  !   │    ② allocate dos 6 campos com dimensão (npt, 1:n) em cada slot   │
  !   │                                                                    │
  !   │  Loop !$omp parallel do (dinâmico por thread):                    │
  !   │    tid = omp_get_thread_num()                                      │
  !   │    call fieldsinfreqs_ws(ws_pool(tid), ...)                       │
  !   │      ├── hmd_TIV_optimized_ws → ws%Tudw, ws%Txdw, ws%Tuup, ws%Txup │
  !   │      └── vmd_optimized_ws    → ws%TEdwz, ws%TEupz                │
  !   │                                                                    │
  !   │  Após o loop (serial):                                            │
  !   │    ③ deallocate dos 6 campos em cada slot                         │
  !   │    ④ deallocate(ws_pool)                                          │
  !   └────────────────────────────────────────────────────────────────────┘
  !
  ! Tabela dos componentes:
  !
  !   ┌─────────────┬──────────────┬─────────────────────────────────────┐
  !   │  Campo      │  Dimensão    │  Semântica                          │
  !   ├─────────────┼──────────────┼─────────────────────────────────────┤
  !   │  Tudw       │  (npt, 1:n)  │  Coef. transmissão TE  — descendente │
  !   │  Txdw       │  (npt, 1:n)  │  Coef. transmissão TM  — descendente │
  !   │  Tuup       │  (npt, 1:n)  │  Coef. transmissão TE  — ascendente  │
  !   │  Txup       │  (npt, 1:n)  │  Coef. transmissão TM  — ascendente  │
  !   │  TEdwz      │  (npt, 1:n)  │  Potencial VMD TE z    — descendente │
  !   │  TEupz      │  (npt, 1:n)  │  Potencial VMD TE z    — ascendente  │
  !   └─────────────┴──────────────┴─────────────────────────────────────┘
  !
  ! Custo de memória estimado (n=10 camadas, npt=201, maxthreads=8):
  !   6 × (201 × 10 × 16 bytes) × 8 threads ≈ 155 KB × 8 ≈ 1,2 MB
  !   (Cabe confortavelmente na L3 de CPUs modernas; nenhum impacto de
  !    cache vs a versão original com allocate/deallocate no hot path.)
  !
  ! Dimensão (npt, 1:n) — justificativa:
  !   Nas rotinas originais, Tudw/Txdw/Tuup/Txup eram dimensionados como
  !   (npt, camadT:camadR), onde [camadT, camadR] ⊆ [1, n] varia por
  !   iteração j. Usar (npt, 1:n) no workspace permite que todos os
  !   índices válidos permaneçam acessíveis sem realloc; os slots fora
  !   do intervalo [camadT, camadR] contêm lixo que nunca é lido pela
  !   lógica de cálculo. Nenhuma mudança matemática é introduzida.
  !
  ! Thread safety:
  !   Cada thread acessa exclusivamente ws_pool(tid), onde tid é obtido
  !   por omp_get_thread_num() dentro da região paralela. Não há aliasing
  !   nem necessidade de locks — cada workspace é privado por construção.
  !
  ! Referência: docs/reference/analise_paralelismo_cpu_fortran.md §7 Fase 3
  !             docs/reference/relatorio_fase3_fortran.md
  !§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
  type :: thread_workspace
    ! ── Fase 3 — arrays de transmissão/potencial (npt × n) ──────────────
    complex(dp), allocatable :: Tudw(:,:)   ! (npt, 1:n) — coef. transmissão TE descendente
    complex(dp), allocatable :: Txdw(:,:)   ! (npt, 1:n) — coef. transmissão TM descendente
    complex(dp), allocatable :: Tuup(:,:)   ! (npt, 1:n) — coef. transmissão TE ascendente
    complex(dp), allocatable :: Txup(:,:)   ! (npt, 1:n) — coef. transmissão TM ascendente
    complex(dp), allocatable :: TEdwz(:,:)  ! (npt, 1:n) — potencial VMD TE z descendente
    complex(dp), allocatable :: TEupz(:,:)  ! (npt, 1:n) — potencial VMD TE z ascendente
    ! ── Fase 3b — fatores de onda de commonfactorsMD (npt) ──────────────
    ! Antes eram automatic arrays em fieldsinfreqs_cached_ws (~19 KB/thread).
    ! Para n ≥ 30 camadas e muitos threads, a pressão de stack acumula-se.
    ! Movidos para heap via workspace para robustez e suporte a n grande.
    complex(dp), allocatable :: Mxdw(:)    ! (npt) — fator reflexão TM descendente
    complex(dp), allocatable :: Mxup(:)    ! (npt) — fator reflexão TM ascendente
    complex(dp), allocatable :: Eudw(:)    ! (npt) — fator reflexão TE descendente
    complex(dp), allocatable :: Euup(:)    ! (npt) — fator reflexão TE ascendente
    complex(dp), allocatable :: FEdwz(:)   ! (npt) — fator TE z-potencial descendente (VMD)
    complex(dp), allocatable :: FEupz(:)   ! (npt) — fator TE z-potencial ascendente (VMD)
  end type thread_workspace

contains
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§��§§§§§§§
subroutine hmd_TIV_optimized(Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, eta, &
                              cx, cy, z, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, Mxdw, Mxup, Eudw, Euup, &
                              Hx_p, Hy_p, Hz_p, dipolo)
  !Atenção!
    ! Esta versão foi adaptada da versão original para compreender apenas saídas de campo magnético, adequadas à perfilagem de poço.
    ! Qualquer alteração/correção será bem vinda se já for realizada no código original, que está na pasta TatuAniso2EMMI, que contém
    ! todos os programas para modelagem eletromagnética de qualquer dipolo, magnético e elétrico, para resistividades com anisotropia TIV.
  ! INPUT:
    ! Tx: abscissa do transmissor
    ! Ty: ordenada do transmissor
    ! h0: cota do transmissor (negativa se acima da primeira interface)
    ! camadR: camada de posição do receptor
    ! camadT: camada de posição do transmissor
    ! npt: número de pontos do filtro para transformada de Hankel
    ! krJ0J1: abscissas do filtro
    ! wJ0: pesos do filtro para a contribuição da Bessel J0
    ! wJ1: pesos do filtro para a contribuição da Bessel J1
    ! h: espessuras das camadas (tem alocação 0:n, com valores em 0 e n para contornar problemas das exponenciais)
    ! prof: profundidades das camadas (tem alocação -1:n usando em -1, 0 e n apenas para contornar problemas com as exponenciais)
    ! zeta: impeditividade
    ! eta: admitividades das n camadas (aqui no caso anisotrópico, são somente as condutividades)
    ! cx: abscissa do receptor
    ! cy: ordenada do receptor
    ! z: cota do receptor
    ! u: constante de propagação horizontal de cada camada
    ! s: constante de propagação vertical (v), multiplicada por lambda (sqrt{\sigma_h/\sigma_v}) (i.e.:  lambda*v)
    ! uh: array de dimensão (npt,n+1) com o produto u*h
    ! sh: array de dimensão (npt,n+1) com o produto (lambda*v) * h
    ! RTEdw: coeficientes de reflexão das camadas inferiores do modo TE
    ! RTEup: coeficientes de reflexão das camadas superiores do modo TE
    ! RTMdw: coeficientes de reflexão das camadas inferiores do modo TM
    ! RTMup: coeficientes de reflexão das camadas superiores do modo TM
    ! Mxdw: fator de onda refletida pelas camadas inferiores do modo TE (potencial \pi_x)
    ! Mxup: fator de onda refletida pelas camadas superiores do modo TE
    ! Eudw: fator de onda refletida pelas camadas inferiores do modo TM (potencial \pi_u)
    ! Euup: fator de onda refletida pelas camadas superiores do modo TM
    ! dipolo: string que designa para que dipolo se deseja obter o campo. Pode ser: hmdx, hmdy ou hmdxy
    ! OUTPUT:
    ! Hx_p, Hy_p, Hz_p: campo magnético primário do DMHx, ou DMHy ou de ambos no domínio espacial
    ! ===================================================================================================================
    ! O DMHy é obtido como rotação (de 90° no sentido anti-horário (de x para y)) do hmdx em torno do eixo z.
    ! As mudanças são as seguintes:
    ! Nas coordenadas: hmdx -----> hmdy
    !                    x  ----->  y
    !                    y  -----> -x
    ! Nas componentes: hmdx -----> hmdy
    !                   Ex  ----->  Ey
    !                  -Ey  ----->  Ex
    !                   Hx  ----->  Hy
    !                  -Hy  ----->  Hx
    ! ===================================================================================================================
  implicit none
  integer,intent(in) :: n, camadT, camadR, npt
  real(dp), intent(in) :: Tx, Ty, h0, h(1:n), prof(0:n), krJ0J1(npt), wJ0(npt), wJ1(npt) 
  real(dp), intent(in) :: cx, cy, z, eta(1:n,2)
  complex(dp), intent(in) :: zeta, Mxdw(npt), Mxup(npt), Eudw(npt), Euup(npt)
  complex(dp), dimension(npt,1:n), intent(in) :: u, s, uh, sh, RTEdw, RTMdw, RTEup, RTMup
  ! complex(dp), dimension(:,:), allocatable, intent(out) :: Hx_p, Hy_p, Hz_p
  complex(dp), dimension(1,2), intent(out) :: Hx_p, Hy_p, Hz_p
  character(5), intent(in) :: dipolo

  integer :: j
  real(dp) :: x, y, r, kr(npt), x2_r2, y2_r2, xy_r2, twox2_r2m1, twoy2_r2m1, twopir
  complex(dp), dimension(npt) :: Ktm, Ktm_J0, Ktm_J1
  complex(dp), dimension(npt) :: Kte, Kte_J1, Ktedz, Ktedz_J0, Ktedz_J1
  complex(dp) :: kh2(1:n), kernelHxJ0, kernelHxJ1, kernelHyJ0, kernelHyJ1, kernelHzJ1
  complex(dp), dimension(:,:), allocatable :: Tudw, Txdw, Tuup, Txup

  if ( dabs(cx - Tx) < eps ) then
    x = 0.d0
  else
    x = cx - Tx
  end if
  if ( dabs(cy - Ty) < eps ) then
    y = 0.d0
  else
    y = cy - Ty
  end if
  r = dsqrt( x ** 2 + y ** 2 )
  if ( r < eps ) r = 1.d-2

  kr = krJ0J1 / r
  ! To workaround the warning: ... may be used uninitialized in this function
  allocate(Tudw(1,1), Txdw(1,1))
  allocate(Tuup(1,1), Txup(1,1))

  ! Rx, Tx e Imp estão associados ao modo TM, enquanto Ru, Tu e Adm estão associados ao modo TE
  if (camadR > camadT) then
    deallocate(Tudw, Txdw)
    allocate(Tudw(npt,camadT:camadR),Txdw(npt,camadT:camadR))
    do j = camadT, camadR
      if (j == camadT) then
        Txdw(:,j)= mx / (2.d0 * s(:,camadT))
        Tudw(:,j)=-mx / 2.d0
      elseif (j == (camadT + 1) .and. j == n) then
        if (n > 1) then
          Txdw(:,j) = s(:,j-1) * Txdw(:,j-1) * (exp(-s(:,j-1) * (prof(camadT)-h0)) + &
                  RTMup(:,j-1) * Mxup * exp(-sh(:,j-1)) - RTMdw(:,j-1) * Mxdw) / s(:,j)
          ! Txdw(:,j) = eta(j-1,1) * Txdw(:,j-1) * (exp(-s(:,j-1) * (prof(camadT)-h0)) + &
          !         RTMup(:,j-1) * Mxup * exp(-sh(:,j-1)) + RTMdw(:,j-1) * Mxdw) / eta(j,1)
        
          Tudw(:,j) = u(:,j-1) * Tudw(:,j-1) * (exp(-u(:,j-1) * (prof(camadT)-h0)) - &
                  RTEup(:,j-1) * Euup * exp(-uh(:,j-1)) - RTEdw(:,j-1) * Eudw) / u(:,j)
          ! Tudw(:,j) = Tudw(:,j-1) * (exp(-u(:,j-1) * (prof(camadT)-h0)) - &
          !         RTEup(:,j-1) * Euup * exp(-uh(:,j-1)) + RTEdw(:,j-1) * Eudw)
        else
          Txdw(:,j) = s(:,j-1) * Txdw(:,j-1) * (exp(s(:,j-1) * h0) - RTMdw(:,j-1) * Mxdw) / s(:,j)
          ! Txdw(:,j) = eta0(1,1) * Txdw(:,j-1) * (exp(s(:,j-1) * h0) + RTMdw(:,j-1) * Mxdw) / eta(j,1)
          ! Txdw(:,j) = eta(j-1,1) * Txdw(:,j-1) * (exp(s(:,j-1) * h0) + RTMdw(:,j-1) * Mxdw) / eta(j,1)
        
          Tudw(:,j) = u(:,j-1) * Tudw(:,j-1) * (exp(u(:,j-1) * h0) - RTEdw(:,j-1) * Eudw) / u(:,j)
          ! Tudw(:,j) = Tudw(:,j-1) * (exp(u(:,j-1) * h0) + RTEdw(:,j-1) * Eudw)
        end if
      elseif (j == (camadT + 1) .and. j /= n) then
        Txdw(:,j) = s(:,j-1) * Txdw(:,j-1) * (exp(-s(:,j-1) * (prof(camadT)-h0)) + &
                RTMup(:,j-1) * Mxup * exp(-sh(:,j-1)) - RTMdw(:,j-1) * Mxdw) / &
                ((1.d0 - RTMdw(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))
        ! Txdw(:,j) = eta(j-1,1) * Txdw(:,j-1) * (exp(-s(:,j-1) * (prof(camadT)-h0)) + &
        !           RTMup(:,j-1) * Mxup * exp(-sh(:,j-1)) + RTMdw(:,j-1) * Mxdw) / &
        !           ((1.d0 + RTMdw(:,j) * exp(-2.d0*sh(:,j))) * eta(j,1))

        Tudw(:,j) = u(:,j-1) * Tudw(:,j-1) * (exp(-u(:,j-1) * (prof(camadT)-h0)) - &
                RTEup(:,j-1) * Euup * exp(-uh(:,j-1)) - RTEdw(:,j-1) * Eudw) / &
                ((1.d0 - RTEdw(:,j) * exp(-2.d0*uh(:,j))) * u(:,j))
        ! Tudw(:,j) = Tudw(:,j-1) * (exp(-u(:,j-1) * (prof(camadT)-h0)) - &
        !         RTEup(:,j-1) * Euup * exp(-uh(:,j-1)) + RTEdw(:,j-1) * Eudw) / &
        !         (1.d0 + RTEdw(:,j) * exp(-2.d0*uh(:,j)))
      elseif (j /= n) then
        Txdw(:,j) = s(:,j-1) * Txdw(:,j-1) * exp(-sh(:,j-1)) * (1.d0 - RTMdw(:,j-1)) / &
                  ((1.d0 - RTMdw(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))
        ! Txdw(:,j) = eta(j-1,1) * Txdw(:,j-1) * exp(-sh(:,j-1)) * (1.d0 + RTMdw(:,j-1)) / &
        !           ((1.d0 + RTMdw(:,j) * exp(-2.d0*sh(:,j))) * eta(j,1))
        
        Tudw(:,j) = u(:,j-1) * Tudw(:,j-1) * exp(-uh(:,j-1)) * (1.d0 - RTEdw(:,j-1))  / &
                  ((1.d0 - RTEdw(:,j) * exp(-2.d0*uh(:,j))) * u(:,j))
        ! Tudw(:,j) = Tudw(:,j-1) * exp(-uh(:,j-1)) * (1.d0 + RTEdw(:,j-1))  / &
        !           (1.d0 + RTEdw(:,j) * exp(-2.d0*uh(:,j)))
      elseif (j == n) then
        Txdw(:,j) = s(:,j-1) * Txdw(:,j-1) * exp(-sh(:,j-1)) * (1.d0 - RTMdw(:,j-1)) / s(:,j)
        ! Txdw(:,j) = eta(j-1,1) * Txdw(:,j-1) * exp(-sh(:,j-1)) * (1.d0 + RTMdw(:,j-1)) / eta(j,1)
        
        Tudw(:,j) = u(:,j-1) * Tudw(:,j-1) * exp(-uh(:,j-1)) * (1.d0 - RTEdw(:,j-1)) / u(:,j)
        ! Tudw(:,j) = Tudw(:,j-1) * exp(-uh(:,j-1)) * (1.d0 + RTEdw(:,j-1))
      end if
    end do
  elseif (camadR < camadT) then
    deallocate(Tuup, Txup)
    allocate(Tuup(npt,camadR:camadT),Txup(npt,camadR:camadT))
    do j = camadT, camadR, -1
      if (j == camadT) then
        Txup(:,j) = mx / (2.d0 * s(:,camadT))
        Tuup(:,j) = mx / 2.d0
      elseif (j == (camadT - 1) .and. j == 1) then
        if (n > 1) then
          Txup(:,j) = s(:,j+1) * Txup(:,j+1) * (exp(-s(:,j+1)*h0) - RTMup(:,j+1) * Mxup + &
                      RTMdw(:,j+1) * Mxdw * exp(-sh(:,j+1))) / s(:,j)
          ! Txup(:,j) = eta(j+1,1) * Txup(:,j+1) * (exp(-s(:,j+1)*h0) + RTMup(:,j+1) * Mxup + &
          !             RTMdw(:,j+1) * Mxdw * exp(-sh(:,j+1))) / eta0(1,1) !eta(j,1)

          Tuup(:,j) = u(:,j+1) * Tuup(:,j+1) * (exp(-u(:,j+1) * h0) - RTEup(:,j+1) * Euup - &
                      RTEdw(:,j+1) * Eudw * exp(-uh(:,j+1))) / u(:,j)
          ! Tuup(:,j) = Tuup(:,j+1) * (exp(-u(:,j+1) * h0) + RTEup(:,j+1) * Euup - &
          !             RTEdw(:,j+1) * Eudw * exp(-uh(:,j+1)))
        else  !dois semiespaços apenas
          Txup(:,j) = s(:,j+1) * Txup(:,j+1) * (exp(-s(:,j+1)*h0) - RTMup(:,j+1) * Mxup) / s(:,j)
          ! Txup(:,j) = eta(j+1,1) * Txup(:,j+1) * (exp(-s(:,j+1)*h0) + RTMup(:,j+1) * Mxup) / eta0(1,1) !eta(j,1)

          Tuup(:,j) = u(:,j+1) * Tuup(:,j+1) * (exp(-u(:,j+1) * h0) - RTEup(:,j+1) * Euup) / u(:,j)
          ! Tuup(:,j) = Tuup(:,j+1) * (exp(-u(:,j+1) * h0) + RTEup(:,j+1) * Euup)
        end if
      elseif (j == (camadT - 1) .and. j /= 1) then
        Txup(:,j) = s(:,j+1) * Txup(:,j+1) * (exp(s(:,j+1)*(prof(j)-h0)) + &
                RTMdw(:,j+1) * Mxdw * exp(-sh(:,j+1)) - RTMup(:,j+1) *  Mxup) / &
                ((1.d0 - RTMup(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))
        ! Txup(:,j) = eta(j+1,1) * Txup(:,j+1) * (exp(s(:,j+1)*(prof(j)-h0)) + &
        !         RTMdw(:,j+1) * Mxdw * exp(-sh(:,j+1)) + RTMup(:,j+1) *  Mxup) / &
        !         ((1.d0 + RTMup(:,j) * exp(-2.d0*sh(:,j))) * eta(j,1))

        Tuup(:,j) = u(:,j+1) * Tuup(:,j+1) * (exp(u(:,j+1)*(prof(camadT-1)-h0)) - &
                RTEup(:,j+1) * Euup - RTEdw(:,j+1) * Eudw * exp(-uh(:,j+1))) / &
                ((1.d0 - RTEup(:,j)*exp(-2.d0*uh(:,j))) * u(:,j))
        ! Tuup(:,j) = Tuup(:,j+1) * (exp(u(:,j+1)*(prof(camadT-1)-h0)) + &
        !         RTEup(:,j+1) * Euup - RTEdw(:,j+1) * Eudw * exp(-uh(:,j+1))) / &
        !         (1.d0 + RTEup(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j /= 1) then
        Txup(:,j) = s(:,j+1) * Txup(:,j+1) * exp(-sh(:,j+1)) * (1.d0 - RTMup(:,j+1)) / &
                 ((1.d0 - RTMup(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))
        ! Txup(:,j) = eta(j+1,1) * Txup(:,j+1) * exp(-sh(:,j+1)) * (1.d0 + RTMup(:,j+1)) / &
        !          ((1.d0 + RTMup(:,j) * exp(-2.d0*sh(:,j))) * eta(j,1))

        Tuup(:,j) = u(:,j+1) * Tuup(:,j+1) * exp(-uh(:,j+1)) * (1.d0 - RTEup(:,j+1)) / &
                 ((1.d0 - RTEup(:,j)*exp(-2.d0*uh(:,j))) * u(:,j))
        ! Tuup(:,j) = Tuup(:,j+1) * exp(-uh(:,j+1)) * (1.d0 + RTEup(:,j+1)) / &
        !          (1.d0 + RTEup(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j == 1) then
        Txup(:,j) = s(:,j+1) * Txup(:,j+1) * exp(-sh(:,j+1)) * (1.d0 - RTMup(:,j+1)) / s(:,j)
        ! Txup(:,j) = eta(1,j+1) * Txup(:,j+1) * exp(-sh(:,j+1) * (1.d0 + RTMup(:,j+1))) / eta(0,j+1) !eta0(1,1)  !

        Tuup(:,j) = u(:,j+1) * Tuup(:,j+1) * exp(-uh(:,j+1)) * (1.d0 - RTEup(:,j+1)) / u(:,j)
        ! Tuup(:,j) = Tuup(:,j+1) * exp(-uh(:,j+1)) * (1.d0 + RTEup(:,j+1))
      end if
    end do
  else
    deallocate(Tudw, Txdw, Tuup, Txup)
    allocate(Tudw(npt,camadT:camadR), Txdw(npt,camadT:camadR))
    allocate(Tuup(npt,camadR:camadT), Txup(npt,camadR:camadT))
    Tudw(:,camadT) =-mx / 2.d0
    Tuup(:,camadT) = mx / 2.d0
    Txdw(:,camadT) = mx / (2.d0 * s(:,camadT))
    Txup(:,camadT) = Txdw(:,camadT)
  end if

  x2_r2 = x * x / (r * r)
  y2_r2 = y * y / (r * r)
  xy_r2 = x * y / (r * r)
  twox2_r2m1 = 2.d0 * x2_r2 - 1.d0
  twoy2_r2m1 = 2.d0 * y2_r2 - 1.d0
  twopir = 2.d0 * pi * r
  kh2(1:n) = -zeta * eta(:,1)
  if (camadR == 1 .and. camadT /= 1) then
    Ktm = Txup(:,1) * exp(s(:,1) * z)
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = Tuup(:,1) * exp(u(:,1) * z)
    Kte_J1 = Kte * wJ1
    Ktedz_J0 = u(:,1) * Kte * wJ0
    Ktedz_J1 = u(:,1) * Kte * wJ1
  elseif (camadR < camadT) then !camada k
    Ktm = Txup(:,camadR) * (exp(s(:,camadR) * (z - prof(camadR))) + &
          RTMup(:,camadR) * exp(-s(:,camadR) * (z - prof(camadR-1) + h(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = Tuup(:,camadR) * (exp(u(:,camadR) * (z - prof(camadR))) + &
          RTEup(:,camadR) * exp(-u(:,camadR) * (z - prof(camadR-1) + h(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = u(:,camadR) * Tuup(:,camadR) * (exp(u(:,camadR) * (z - prof(camadR))) - &
          RTEup(:,camadR) * exp(-u(:,camadR) * (z - prof(camadR-1) + h(camadR))))
    ktedz_J0 = Ktedz * wJ0
    ktedz_J1 = Ktedz * wJ1
  elseif (camadR == camadT .and. z <= h0) then !na mesma camada do transmissor mas acima dele
    Ktm = Txup(:,camadR) * (exp(s(:,camadR) * (z - h0)) + &
          RTMup(:,camadR) * Mxup * exp(-s(:,camadR) * (z - prof(camadR-1))) + &
          RTMdw(:,camadR) * Mxdw * exp(s(:,camadR) * (z - prof(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = Tuup(:,camadR) * (exp(u(:,camadR) * (z - h0)) + &
          RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) - &
          RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = u(:,camadR) * Tuup(:,camadR) * (exp(u(:,camadR) * (z - h0)) - &
          RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) - &
          RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1
  elseif (camadR == camadT .and. z > h0) then  !na mesma camada do transmissor mas abaixo dele
    Ktm = Txdw(:,camadR) * (exp(-s(:,camadR) * (z - h0)) + &
          RTMup(:,camadR) * Mxup * exp(-s(:,camadR) * (z - prof(camadR-1))) + &
          RTMdw(:,camadR) * Mxdw * exp(s(:,camadR) * (z - prof(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = Tudw(:,camadR) * (exp(-u(:,camadR) * (z - h0)) - &
        RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) + &
        RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = -u(:,camadR) * Tudw(:,camadR) * (exp(-u(:,camadR) * (z - h0)) - &
        RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) - &
        RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1
  elseif (camadR > camadT .and. camadR /= n) then !camada j
    Ktm = Txdw(:,camadR) * (exp(-s(:,camadR) * (z - prof(camadR-1))) + &
          RTMdw(:,camadR) * exp(s(:,camadR) * (z - prof(camadR) - h(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = Tudw(:,camadR) * (exp(-u(:,camadR) * (z - prof(camadR-1))) + &
          RTEdw(:,camadR) * exp(u(:,camadR) * (z - prof(camadR) - h(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = -u(:,camadR) * Tudw(:,camadR) * (exp(-u(:,camadR) * (z - prof(camadR-1))) - &
          RTEdw(:,camadR) * exp(u(:,camadR) * (z - prof(camadR) - h(camadR))))
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1
  else  !camada n
    Ktm = Txdw(:,camadR) * exp(-s(:,camadR) * (z - prof(camadR-1)))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = Tudw(:,camadR) * exp(-u(:,camadR) * (z - prof(camadR-1)))
    Kte_J1 = Kte * wJ1
    Ktedz_J0 = -u(:,camadR) * Kte * wJ0
    Ktedz_J1 = -u(:,camadR) * Kte * wJ1
  end if

  select case(dipolo)
    case('hmdx')
      ! allocate(Hx_p(1,1), Hy_p(1,1), Hz_p(1,1))
      kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r
      kernelHxJ0 = x2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * y2_r2 * sum(Ktm_J0 * kr)
      Hx_p(1,1) = (kernelHxJ1 - kernelHxJ0) / twopir  !o último r é decorrente do uso dos filtros

      kernelHyJ1 = sum(Ktedz_J1 + kh2(camadR) * Ktm_J1) / r
      kernelHyJ0 = sum((Ktedz_J0 + kh2(camadR) * Ktm_J0) * kr) / 2.d0
      Hy_p(1,1) = xy_r2 * (kernelHyJ1 - kernelHyJ0) / pi / r  !o último r é decorrente do uso dos filtros

      kernelHzJ1 = x * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,1) = -kernelHzJ1 / twopir !o último r é decorrente do uso dos filtros
    case('hmdy')
      ! allocate(Hx_p(1,1), Hy_p(1,1), Hz_p(1,1))
      kernelHxJ1 = sum(Ktedz_J1 + kh2(camadR) * Ktm_J1) / r
      kernelHxJ0 = sum((Ktedz_J0 + kh2(camadR) * Ktm_J0) * kr) / 2.d0
      Hx_p(1,1) = xy_r2 * (kernelHxJ1 - kernelHxJ0) / pi / r  !o último r é decorrente do uso dos filtros

      kernelHyJ1 = (twoy2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twox2_r2m1 * sum(Ktm_J1)) / r
      kernelHyJ0 = y2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * x2_r2 * sum(Ktm_J0 * kr)
      Hy_p(1,1) = (kernelHyJ1 - kernelHyJ0) / twopir  !o último r é decorrente do uso dos filtros

      kernelHzJ1 = y * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,1) = -kernelHzJ1 / twopir !o último r é decorrente do uso dos filtros
    case('hmdxy')
      ! allocate(Hx_p(1,2), Hy_p(1,2), Hz_p(1,2))
      kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r
      kernelHxJ0 = x2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * y2_r2 * sum(Ktm_J0 * kr)
      Hx_p(1,1) = (kernelHxJ1 - kernelHxJ0) / twopir  !o último r é decorrente do uso dos filtros

      kernelHyJ1 = sum(Ktedz_J1 + kh2(camadR) * Ktm_J1) / r
      kernelHyJ0 = sum((Ktedz_J0 + kh2(camadR) * Ktm_J0) * kr) / 2.d0
      Hy_p(1,1) = xy_r2 * (kernelHyJ1 - kernelHyJ0) / pi / r  !o último r é decorrente do uso dos filtros

      kernelHzJ1 = x * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,1) = -kernelHzJ1 / twopir !o último r é decorrente do uso dos filtros

      ! kernelHxJ1 = sum(Ktedz_J1 + kh2(camadR) * Ktm_J1) / r
      ! kernelHxJ0 = sum((Ktedz_J0 + kh2(camadR) * Ktm_J0) * kr) / 2.0
      Hx_p(1,2) = Hy_p(1,1)  !xy_r2 * (kernelHxJ1 - kernelHxJ0) / pi / r  !o último r é decorrente do uso dos filtros

      kernelHyJ1 = (twoy2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twox2_r2m1 * sum(Ktm_J1)) / r
      kernelHyJ0 = y2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * x2_r2 * sum(Ktm_J0 * kr)
      Hy_p(1,2) = (kernelHyJ1 - kernelHyJ0) / twopir  !o último r é decorrente do uso dos filtros

      kernelHzJ1 = y * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,2) = -kernelHzJ1 / twopir !o último r é decorrente do uso dos filtros
    case default
      write(*,'(A,A)')' Não se tem programado nenhum dipolo do tipo informado:',dipolo
      stop
    end select
end subroutine hmd_TIV_optimized
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine vmd_optimized(Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, cx, cy, z, &
                         u, uh, AdmInt, RTEdw, RTEup, FEdwz, FEupz, Hx_p, Hy_p, Hz_p)

  implicit none
  integer, intent(in) :: n, camadT, camadR, npt
  real(dp), intent(in) :: Tx, Ty, h0, h(1:n), prof(0:n), krJ0J1(npt), wJ0(npt), wJ1(npt), cx, cy, z
  complex(dp), intent(in) :: zeta, FEdwz(npt), FEupz(npt)
  complex(dp), dimension(npt,1:n), intent(in) :: u, uh, AdmInt, RTEdw, RTEup
  complex(dp), intent(out) :: Hx_p, Hy_p, Hz_p
  ! complex(dp), dimension(:,:), allocatable, intent(out) :: Hx_p, Hy_p, Hz_p

  integer :: j
  real(dp) :: x, y, r, kr(npt), twopir

  complex(dp), dimension(npt) :: fac, KtezJ0, KtezJ1, KtedzzJ1
  complex(dp) :: kernelHx, kernelHy, kernelHz
  complex(dp), dimension(:,:), allocatable :: TEdwz, TEupz

  ! allocate(Hx_p(1,1), Hy_p(1,1), Hz_p(1,1))

  if ( dabs(cx - Tx) < eps ) then
    x = 0.d0
  else
    x = cx - Tx
  end if
  if ( dabs(cy - Ty) < eps ) then
    y = 0.d0
  else
    y = cy - Ty
  end if
  r = dsqrt( x ** 2 + y ** 2 )
  if ( r < eps ) r = 1.d-2

  kr = krJ0J1 / r

  ! To workaround the warning: ... may be used uninitialized in this function
  allocate(TEdwz(1,1), TEupz(1,1))
  if (camadR > camadT) then
    deallocate(TEdwz)
    allocate(TEdwz(npt,camadT:camadR))
    do j = camadT, camadR
      if (j == camadT) then
        TEdwz(:,j) = zeta * mz / ( 2.d0 * u(:,j) )
      elseif (j == (camadT + 1) .and. j == n) then
        TEdwz(:,j) = TEdwz(:,j-1)*(exp(-u(:,camadT)*(prof(camadT)-h0)) + &
          RTEup(:,camadT)*FEupz*exp(-uh(:,camadT))+RTEdw(:,camadT)*FEdwz)
      elseif (j == (camadT + 1) .and. j /= n) then
        TEdwz(:,j) = TEdwz(:,j-1)*(exp(-u(:,camadT)*(prof(camadT)-h0)) + &
          RTEup(:,camadT)*FEupz(:)*exp(-uh(:,camadT)) + &
          RTEdw(:,camadT)*FEdwz(:))/(1.d0+RTEdw(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j /= n) then
        TEdwz(:,j) = TEdwz(:,j-1)*(1.d0+RTEdw(:,j-1))*exp(-uh(:,j-1)) / &
          (1.d0+RTEdw(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j==n) then
        TEdwz(:,j) = TEdwz(:,j-1)*(1.d0+RTEdw(:,j-1))*exp(-uh(:,j-1))
      end if
    end do
  elseif (camadR < camadT) then
    deallocate(TEupz)
    allocate(TEupz(npt,camadR:camadT))
    do j=camadT,camadR,-1
      if (j == camadT) then
        TEupz(:,j) = zeta * mz / (2.d0 * u(:,j))
      elseif (j == (camadT - 1) .and. j == 1) then
        TEupz(:,j) = TEupz(:,j+1)*(exp(-u(:,camadT)*h0) + &
          RTEup(:,camadT)*FEupz(:)+RTEdw(:,camadT)*FEdwz(:)*exp(-uh(:,camadT)))
      elseif (j == (camadT - 1) .and. j /= 1) then
        TEupz(:,j) = TEupz(:,j+1)*(exp(u(:,camadT)*(prof(camadT-1)-h0)) + &
          RTEup(:,camadT)*FEupz(:)+RTEdw(:,camadT)*FEdwz(:) * &
          exp(-uh(:,camadT)))/(1.d0+RTEup(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j /= 1) then
        TEupz(:,j) = TEupz(:,j+1)*(1.d0+RTEup(:,j+1))*exp(-uh(:,j+1)) / &
          (1.d0+RTEup(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j == 1) then
        TEupz(:,j) = TEupz(:,j+1) * (1.d0+RTEup(:,j+1))*exp(-uh(:,j+1))
      end if
    end do
  else
    deallocate(TEdwz, TEupz)
    allocate(TEdwz(npt,camadT:camadR), TEupz(npt,camadR:camadT))
    TEdwz(:,camadR) = zeta * mz / (2.d0 * u(:,camadT))
    TEupz(:,camadR) = TEdwz(:,camadR)
  end if

  twopir = 2.d0 * pi * r
  if (camadR == 1 .and. camadT /= 1) then
    fac = TEupz(:,1)*exp(u(:,1)*z)
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = AdmInt(:,1) * KtezJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r !o último r é decorrente do uso dos filtros

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r !o último r é decorrente do uso dos filtros

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r !o último r é decorrente do uso dos filtros
  elseif (camadR < camadT) then !camada k
    fac = TEupz(:,camadR)*(exp(u(:,camadR)*(z-prof(camadR))) + &
            RTEup(:,camadR)*exp(-u(:,camadR)*(z-prof(camadR-1)+h(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (AdmInt(:,camadR) * TEupz(:,camadR)*(exp(u(:,camadR)*(z-prof(camadR))) - &
      RTEup(:,camadR)*exp(-u(:,camadR)*(z-prof(camadR-1)+h(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r !o último r é decorrente do uso dos filtros

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r !o último r é decorrente do uso dos filtros

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r !o último r é decorrente do uso dos filtros
  elseif (camadR == camadT .and. z <= h0) then !na mesma camada do transmissor mas acima dele
    fac = TEupz(:,camadR)*(exp(u(:,camadR)*(z-h0)) + &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (AdmInt(:,camadR)*TEupz(:,camadR)*(exp(u(:,camadR)*(z-h0)) - &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r !o último r é decorrente do uso dos filtros

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r !o último r é decorrente do uso dos filtros

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r !o último r é decorrente do uso dos filtros
  elseif (camadR == camadT .and. z > h0) then !na mesma camada do transmissor mas abaixo dele
    fac = TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-h0)) + &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (-AdmInt(:,camadR)*TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-h0)) + &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) - &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r !o último r é decorrente do uso dos filtros

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r !o último r é decorrente do uso dos filtros

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r !o último r é decorrente do uso dos filtros
  elseif (camadR > camadT .and. camadR /= n) then !camada j
    fac = TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*exp(u(:,camadR)*(z-prof(camadR)-h(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (-AdmInt(:,camadR)*TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-prof(camadR-1))) - &
      RTEdw(:,camadR)*exp(u(:,camadR)*(z-prof(camadR)-h(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r !o último r é decorrente do uso dos filtros

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r !o último r é decorrente do uso dos filtros

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r !o último r é decorrente do uso dos filtros
  else !camada n
    fac = TEdwz(:,n)*exp(-u(:,n)*(z-prof(n-1)))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = -AdmInt(:,n) * fac * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r !o último r é decorrente do uso dos filtros

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r !o último r é decorrente do uso dos filtros

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r  !o último r é decorrente do uso dos filtros
  end if
end subroutine vmd_optimized
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Fase 3 — Workspace Pre-allocation
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
!
! Variante de hmd_TIV_optimized que recebe um thread_workspace pré-alocado
! em vez de alocar/desalocar internamente Tudw, Txdw, Tuup, Txup. A lógica
! matemática é idêntica à da rotina original — apenas os allocate/deallocate
! dinâmicos no hot path paralelo são eliminados.
!
! Ganho esperado: remoção de 4 allocate/deallocate por chamada × 2 chamadas
! por medida (nf=2) × ~600 medidas = ~4.800 chamadas de malloc/free evitadas
! por modelo. Em 8 threads paralelas, elimina a contenção no mutex do heap.
!
! Preservação da rotina original hmd_TIV_optimized: intencional para permitir
! rollback instantâneo e validação diferencial em caso de regressão numérica.
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine hmd_TIV_optimized_ws(ws, Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, eta, &
                                 cx, cy, z, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, Mxdw, Mxup, Eudw, Euup, &
                                 Hx_p, Hy_p, Hz_p, dipolo)
  ! INPUT:
  !   ws: workspace pré-alocado por thread (campos Tudw, Txdw, Tuup, Txup de dimensão (npt, 1:n)).
  !       As demais campos (TEdwz, TEupz) não são usados nesta rotina.
  !   Demais argumentos: idênticos a hmd_TIV_optimized.
  ! OUTPUT:
  !   Hx_p, Hy_p, Hz_p: campo magnético primário (idêntico à rotina original).
  implicit none
  type(thread_workspace), intent(inout) :: ws
  integer,intent(in) :: n, camadT, camadR, npt
  ! Débito B7 corrigido: contiguous attribute adicionado aos dummy arguments
  ! que recebem slices de arrays maiores (e.g. krwJ0J1(:,1), u_cache(:,:,i)).
  ! Em Fortran 2008, contiguous garante que o compilador não gere cópia
  ! temporária para slices que já são contíguas em memória (column-major).
  real(dp), intent(in) :: Tx, Ty, h0, h(1:n), prof(0:n)
  real(dp), intent(in), contiguous :: krJ0J1(:), wJ0(:), wJ1(:)
  real(dp), intent(in) :: cx, cy, z, eta(1:n,2)
  complex(dp), intent(in) :: zeta
  complex(dp), intent(in), contiguous :: Mxdw(:), Mxup(:), Eudw(:), Euup(:)
  complex(dp), dimension(npt,1:n), intent(in) :: u, s, uh, sh, RTEdw, RTMdw, RTEup, RTMup
  complex(dp), dimension(1,2), intent(out) :: Hx_p, Hy_p, Hz_p
  character(5), intent(in) :: dipolo

  integer :: j
  real(dp) :: x, y, r, kr(npt), x2_r2, y2_r2, xy_r2, twox2_r2m1, twoy2_r2m1, twopir
  complex(dp), dimension(npt) :: Ktm, Ktm_J0, Ktm_J1
  complex(dp), dimension(npt) :: Kte, Kte_J1, Ktedz, Ktedz_J0, Ktedz_J1
  complex(dp) :: kh2(1:n), kernelHxJ0, kernelHxJ1, kernelHyJ0, kernelHyJ1, kernelHzJ1

  if ( dabs(cx - Tx) < eps ) then
    x = 0.d0
  else
    x = cx - Tx
  end if
  if ( dabs(cy - Ty) < eps ) then
    y = 0.d0
  else
    y = cy - Ty
  end if
  r = dsqrt( x ** 2 + y ** 2 )
  if ( r < eps ) r = 1.d-2

  kr = krJ0J1 / r

  ! Rx, Tx e Imp estão associados ao modo TM, enquanto Ru, Tu e Adm estão associados ao modo TE
  if (camadR > camadT) then
    do j = camadT, camadR
      if (j == camadT) then
        ws%Txdw(:,j)= mx / (2.d0 * s(:,camadT))
        ws%Tudw(:,j)=-mx / 2.d0
      elseif (j == (camadT + 1) .and. j == n) then
        if (n > 1) then
          ws%Txdw(:,j) = s(:,j-1) * ws%Txdw(:,j-1) * (exp(-s(:,j-1) * (prof(camadT)-h0)) + &
                  RTMup(:,j-1) * Mxup * exp(-sh(:,j-1)) - RTMdw(:,j-1) * Mxdw) / s(:,j)

          ws%Tudw(:,j) = u(:,j-1) * ws%Tudw(:,j-1) * (exp(-u(:,j-1) * (prof(camadT)-h0)) - &
                  RTEup(:,j-1) * Euup * exp(-uh(:,j-1)) - RTEdw(:,j-1) * Eudw) / u(:,j)
        else
          ws%Txdw(:,j) = s(:,j-1) * ws%Txdw(:,j-1) * (exp(s(:,j-1) * h0) - RTMdw(:,j-1) * Mxdw) / s(:,j)

          ws%Tudw(:,j) = u(:,j-1) * ws%Tudw(:,j-1) * (exp(u(:,j-1) * h0) - RTEdw(:,j-1) * Eudw) / u(:,j)
        end if
      elseif (j == (camadT + 1) .and. j /= n) then
        ws%Txdw(:,j) = s(:,j-1) * ws%Txdw(:,j-1) * (exp(-s(:,j-1) * (prof(camadT)-h0)) + &
                RTMup(:,j-1) * Mxup * exp(-sh(:,j-1)) - RTMdw(:,j-1) * Mxdw) / &
                ((1.d0 - RTMdw(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))

        ws%Tudw(:,j) = u(:,j-1) * ws%Tudw(:,j-1) * (exp(-u(:,j-1) * (prof(camadT)-h0)) - &
                RTEup(:,j-1) * Euup * exp(-uh(:,j-1)) - RTEdw(:,j-1) * Eudw) / &
                ((1.d0 - RTEdw(:,j) * exp(-2.d0*uh(:,j))) * u(:,j))
      elseif (j /= n) then
        ws%Txdw(:,j) = s(:,j-1) * ws%Txdw(:,j-1) * exp(-sh(:,j-1)) * (1.d0 - RTMdw(:,j-1)) / &
                  ((1.d0 - RTMdw(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))

        ws%Tudw(:,j) = u(:,j-1) * ws%Tudw(:,j-1) * exp(-uh(:,j-1)) * (1.d0 - RTEdw(:,j-1))  / &
                  ((1.d0 - RTEdw(:,j) * exp(-2.d0*uh(:,j))) * u(:,j))
      elseif (j == n) then
        ws%Txdw(:,j) = s(:,j-1) * ws%Txdw(:,j-1) * exp(-sh(:,j-1)) * (1.d0 - RTMdw(:,j-1)) / s(:,j)

        ws%Tudw(:,j) = u(:,j-1) * ws%Tudw(:,j-1) * exp(-uh(:,j-1)) * (1.d0 - RTEdw(:,j-1)) / u(:,j)
      end if
    end do
  elseif (camadR < camadT) then
    do j = camadT, camadR, -1
      if (j == camadT) then
        ws%Txup(:,j) = mx / (2.d0 * s(:,camadT))
        ws%Tuup(:,j) = mx / 2.d0
      elseif (j == (camadT - 1) .and. j == 1) then
        if (n > 1) then
          ws%Txup(:,j) = s(:,j+1) * ws%Txup(:,j+1) * (exp(-s(:,j+1)*h0) - RTMup(:,j+1) * Mxup + &
                      RTMdw(:,j+1) * Mxdw * exp(-sh(:,j+1))) / s(:,j)

          ws%Tuup(:,j) = u(:,j+1) * ws%Tuup(:,j+1) * (exp(-u(:,j+1) * h0) - RTEup(:,j+1) * Euup - &
                      RTEdw(:,j+1) * Eudw * exp(-uh(:,j+1))) / u(:,j)
        else  !dois semiespaços apenas
          ws%Txup(:,j) = s(:,j+1) * ws%Txup(:,j+1) * (exp(-s(:,j+1)*h0) - RTMup(:,j+1) * Mxup) / s(:,j)

          ws%Tuup(:,j) = u(:,j+1) * ws%Tuup(:,j+1) * (exp(-u(:,j+1) * h0) - RTEup(:,j+1) * Euup) / u(:,j)
        end if
      elseif (j == (camadT - 1) .and. j /= 1) then
        ws%Txup(:,j) = s(:,j+1) * ws%Txup(:,j+1) * (exp(s(:,j+1)*(prof(j)-h0)) + &
                RTMdw(:,j+1) * Mxdw * exp(-sh(:,j+1)) - RTMup(:,j+1) *  Mxup) / &
                ((1.d0 - RTMup(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))

        ws%Tuup(:,j) = u(:,j+1) * ws%Tuup(:,j+1) * (exp(u(:,j+1)*(prof(camadT-1)-h0)) - &
                RTEup(:,j+1) * Euup - RTEdw(:,j+1) * Eudw * exp(-uh(:,j+1))) / &
                ((1.d0 - RTEup(:,j)*exp(-2.d0*uh(:,j))) * u(:,j))
      elseif (j /= 1) then
        ws%Txup(:,j) = s(:,j+1) * ws%Txup(:,j+1) * exp(-sh(:,j+1)) * (1.d0 - RTMup(:,j+1)) / &
                 ((1.d0 - RTMup(:,j) * exp(-2.d0*sh(:,j))) * s(:,j))

        ws%Tuup(:,j) = u(:,j+1) * ws%Tuup(:,j+1) * exp(-uh(:,j+1)) * (1.d0 - RTEup(:,j+1)) / &
                 ((1.d0 - RTEup(:,j)*exp(-2.d0*uh(:,j))) * u(:,j))
      elseif (j == 1) then
        ws%Txup(:,j) = s(:,j+1) * ws%Txup(:,j+1) * exp(-sh(:,j+1)) * (1.d0 - RTMup(:,j+1)) / s(:,j)

        ws%Tuup(:,j) = u(:,j+1) * ws%Tuup(:,j+1) * exp(-uh(:,j+1)) * (1.d0 - RTEup(:,j+1)) / u(:,j)
      end if
    end do
  else
    ws%Tudw(:,camadT) =-mx / 2.d0
    ws%Tuup(:,camadT) = mx / 2.d0
    ws%Txdw(:,camadT) = mx / (2.d0 * s(:,camadT))
    ws%Txup(:,camadT) = ws%Txdw(:,camadT)
  end if

  x2_r2 = x * x / (r * r)
  y2_r2 = y * y / (r * r)
  xy_r2 = x * y / (r * r)
  twox2_r2m1 = 2.d0 * x2_r2 - 1.d0
  twoy2_r2m1 = 2.d0 * y2_r2 - 1.d0
  twopir = 2.d0 * pi * r
  kh2(1:n) = -zeta * eta(:,1)
  if (camadR == 1 .and. camadT /= 1) then
    Ktm = ws%Txup(:,1) * exp(s(:,1) * z)
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = ws%Tuup(:,1) * exp(u(:,1) * z)
    Kte_J1 = Kte * wJ1
    Ktedz_J0 = u(:,1) * Kte * wJ0
    Ktedz_J1 = u(:,1) * Kte * wJ1
  elseif (camadR < camadT) then !camada k
    Ktm = ws%Txup(:,camadR) * (exp(s(:,camadR) * (z - prof(camadR))) + &
          RTMup(:,camadR) * exp(-s(:,camadR) * (z - prof(camadR-1) + h(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = ws%Tuup(:,camadR) * (exp(u(:,camadR) * (z - prof(camadR))) + &
          RTEup(:,camadR) * exp(-u(:,camadR) * (z - prof(camadR-1) + h(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = u(:,camadR) * ws%Tuup(:,camadR) * (exp(u(:,camadR) * (z - prof(camadR))) - &
          RTEup(:,camadR) * exp(-u(:,camadR) * (z - prof(camadR-1) + h(camadR))))
    ktedz_J0 = Ktedz * wJ0
    ktedz_J1 = Ktedz * wJ1
  elseif (camadR == camadT .and. z <= h0) then !na mesma camada do transmissor mas acima dele
    Ktm = ws%Txup(:,camadR) * (exp(s(:,camadR) * (z - h0)) + &
          RTMup(:,camadR) * Mxup * exp(-s(:,camadR) * (z - prof(camadR-1))) + &
          RTMdw(:,camadR) * Mxdw * exp(s(:,camadR) * (z - prof(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = ws%Tuup(:,camadR) * (exp(u(:,camadR) * (z - h0)) + &
          RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) - &
          RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = u(:,camadR) * ws%Tuup(:,camadR) * (exp(u(:,camadR) * (z - h0)) - &
          RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) - &
          RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1
  elseif (camadR == camadT .and. z > h0) then  !na mesma camada do transmissor mas abaixo dele
    Ktm = ws%Txdw(:,camadR) * (exp(-s(:,camadR) * (z - h0)) + &
          RTMup(:,camadR) * Mxup * exp(-s(:,camadR) * (z - prof(camadR-1))) + &
          RTMdw(:,camadR) * Mxdw * exp(s(:,camadR) * (z - prof(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = ws%Tudw(:,camadR) * (exp(-u(:,camadR) * (z - h0)) - &
        RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) + &
        RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = -u(:,camadR) * ws%Tudw(:,camadR) * (exp(-u(:,camadR) * (z - h0)) - &
        RTEup(:,camadR) * Euup * exp(-u(:,camadR) * (z - prof(camadR-1))) - &
        RTEdw(:,camadR) * Eudw * exp(u(:,camadR) * (z - prof(camadR))))
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1
  elseif (camadR > camadT .and. camadR /= n) then !camada j
    Ktm = ws%Txdw(:,camadR) * (exp(-s(:,camadR) * (z - prof(camadR-1))) + &
          RTMdw(:,camadR) * exp(s(:,camadR) * (z - prof(camadR) - h(camadR))))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = ws%Tudw(:,camadR) * (exp(-u(:,camadR) * (z - prof(camadR-1))) + &
          RTEdw(:,camadR) * exp(u(:,camadR) * (z - prof(camadR) - h(camadR))))
    Kte_J1 = Kte * wJ1
    Ktedz = -u(:,camadR) * ws%Tudw(:,camadR) * (exp(-u(:,camadR) * (z - prof(camadR-1))) - &
          RTEdw(:,camadR) * exp(u(:,camadR) * (z - prof(camadR) - h(camadR))))
    Ktedz_J0 = Ktedz * wJ0
    Ktedz_J1 = Ktedz * wJ1
  else  !camada n
    Ktm = ws%Txdw(:,camadR) * exp(-s(:,camadR) * (z - prof(camadR-1)))
    Ktm_J0 = Ktm * wJ0
    Ktm_J1 = Ktm * wJ1

    Kte = ws%Tudw(:,camadR) * exp(-u(:,camadR) * (z - prof(camadR-1)))
    Kte_J1 = Kte * wJ1
    Ktedz_J0 = -u(:,camadR) * Kte * wJ0
    Ktedz_J1 = -u(:,camadR) * Kte * wJ1
  end if

  select case(dipolo)
    case('hmdx')
      kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r
      kernelHxJ0 = x2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * y2_r2 * sum(Ktm_J0 * kr)
      Hx_p(1,1) = (kernelHxJ1 - kernelHxJ0) / twopir

      kernelHyJ1 = sum(Ktedz_J1 + kh2(camadR) * Ktm_J1) / r
      kernelHyJ0 = sum((Ktedz_J0 + kh2(camadR) * Ktm_J0) * kr) / 2.d0
      Hy_p(1,1) = xy_r2 * (kernelHyJ1 - kernelHyJ0) / pi / r

      kernelHzJ1 = x * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,1) = -kernelHzJ1 / twopir
    case('hmdy')
      kernelHxJ1 = sum(Ktedz_J1 + kh2(camadR) * Ktm_J1) / r
      kernelHxJ0 = sum((Ktedz_J0 + kh2(camadR) * Ktm_J0) * kr) / 2.d0
      Hx_p(1,1) = xy_r2 * (kernelHxJ1 - kernelHxJ0) / pi / r

      kernelHyJ1 = (twoy2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twox2_r2m1 * sum(Ktm_J1)) / r
      kernelHyJ0 = y2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * x2_r2 * sum(Ktm_J0 * kr)
      Hy_p(1,1) = (kernelHyJ1 - kernelHyJ0) / twopir

      kernelHzJ1 = y * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,1) = -kernelHzJ1 / twopir
    case('hmdxy')
      kernelHxJ1 = (twox2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twoy2_r2m1 * sum(Ktm_J1)) / r
      kernelHxJ0 = x2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * y2_r2 * sum(Ktm_J0 * kr)
      Hx_p(1,1) = (kernelHxJ1 - kernelHxJ0) / twopir

      kernelHyJ1 = sum(Ktedz_J1 + kh2(camadR) * Ktm_J1) / r
      kernelHyJ0 = sum((Ktedz_J0 + kh2(camadR) * Ktm_J0) * kr) / 2.d0
      Hy_p(1,1) = xy_r2 * (kernelHyJ1 - kernelHyJ0) / pi / r

      kernelHzJ1 = x * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,1) = -kernelHzJ1 / twopir

      Hx_p(1,2) = Hy_p(1,1)

      kernelHyJ1 = (twoy2_r2m1 * sum(Ktedz_J1) - kh2(camadR) * twox2_r2m1 * sum(Ktm_J1)) / r
      kernelHyJ0 = y2_r2 * sum(Ktedz_J0 * kr) - kh2(camadR) * x2_r2 * sum(Ktm_J0 * kr)
      Hy_p(1,2) = (kernelHyJ1 - kernelHyJ0) / twopir

      kernelHzJ1 = y * sum(Kte_J1 * kr * kr) / r
      Hz_p(1,2) = -kernelHzJ1 / twopir
    case default
      write(*,'(A,A)')' Não se tem programado nenhum dipolo do tipo informado:',dipolo
      stop
    end select
end subroutine hmd_TIV_optimized_ws
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Fase 3 — Workspace Pre-allocation
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
!
! Variante de vmd_optimized que recebe um thread_workspace pré-alocado em vez
! de alocar/desalocar internamente TEdwz e TEupz. A lógica matemática é
! idêntica à da rotina original — apenas os allocate/deallocate dinâmicos no
! hot path paralelo são eliminados.
!
! Ganho esperado: remoção de 2 allocate/deallocate por chamada × 2 chamadas
! por medida (nf=2) × ~600 medidas = ~2.400 chamadas de malloc/free evitadas
! por modelo. Somadas às ~4.800 eliminadas em hmd_TIV_optimized_ws, completam
! a eliminação total do malloc no hot path paralelo da Fase 3.
!
! Preservação da rotina original vmd_optimized: intencional, análoga à
! preservação de hmd_TIV_optimized, para rollback e validação diferencial.
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine vmd_optimized_ws(ws, Tx, Ty, h0, n, camadR, camadT, npt, krJ0J1, wJ0, wJ1, h, prof, zeta, cx, cy, z, &
                            u, uh, AdmInt, RTEdw, RTEup, FEdwz, FEupz, Hx_p, Hy_p, Hz_p)
  ! INPUT:
  !   ws: workspace pré-alocado por thread (campos TEdwz, TEupz de dimensão (npt, 1:n)).
  !       Os campos Tudw, Txdw, Tuup, Txup não são usados nesta rotina.
  !   Demais argumentos: idênticos a vmd_optimized.
  ! OUTPUT:
  !   Hx_p, Hy_p, Hz_p: campo magnético primário do VMD (idêntico à rotina original).
  implicit none
  type(thread_workspace), intent(inout) :: ws
  integer, intent(in) :: n, camadT, camadR, npt
  ! Débito B7 corrigido: contiguous attribute em dummy arguments de slices.
  real(dp), intent(in) :: Tx, Ty, h0, h(1:n), prof(0:n)
  real(dp), intent(in), contiguous :: krJ0J1(:), wJ0(:), wJ1(:)
  real(dp), intent(in) :: cx, cy, z
  complex(dp), intent(in) :: zeta
  complex(dp), intent(in), contiguous :: FEdwz(:), FEupz(:)
  complex(dp), dimension(npt,1:n), intent(in) :: u, uh, AdmInt, RTEdw, RTEup
  complex(dp), intent(out) :: Hx_p, Hy_p, Hz_p

  integer :: j
  real(dp) :: x, y, r, kr(npt), twopir
  complex(dp), dimension(npt) :: fac, KtezJ0, KtezJ1, KtedzzJ1
  complex(dp) :: kernelHx, kernelHy, kernelHz

  if ( dabs(cx - Tx) < eps ) then
    x = 0.d0
  else
    x = cx - Tx
  end if
  if ( dabs(cy - Ty) < eps ) then
    y = 0.d0
  else
    y = cy - Ty
  end if
  r = dsqrt( x ** 2 + y ** 2 )
  if ( r < eps ) r = 1.d-2

  kr = krJ0J1 / r

  if (camadR > camadT) then
    do j = camadT, camadR
      if (j == camadT) then
        ws%TEdwz(:,j) = zeta * mz / ( 2.d0 * u(:,j) )
      elseif (j == (camadT + 1) .and. j == n) then
        ws%TEdwz(:,j) = ws%TEdwz(:,j-1)*(exp(-u(:,camadT)*(prof(camadT)-h0)) + &
          RTEup(:,camadT)*FEupz*exp(-uh(:,camadT))+RTEdw(:,camadT)*FEdwz)
      elseif (j == (camadT + 1) .and. j /= n) then
        ws%TEdwz(:,j) = ws%TEdwz(:,j-1)*(exp(-u(:,camadT)*(prof(camadT)-h0)) + &
          RTEup(:,camadT)*FEupz(:)*exp(-uh(:,camadT)) + &
          RTEdw(:,camadT)*FEdwz(:))/(1.d0+RTEdw(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j /= n) then
        ws%TEdwz(:,j) = ws%TEdwz(:,j-1)*(1.d0+RTEdw(:,j-1))*exp(-uh(:,j-1)) / &
          (1.d0+RTEdw(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j==n) then
        ws%TEdwz(:,j) = ws%TEdwz(:,j-1)*(1.d0+RTEdw(:,j-1))*exp(-uh(:,j-1))
      end if
    end do
  elseif (camadR < camadT) then
    do j=camadT,camadR,-1
      if (j == camadT) then
        ws%TEupz(:,j) = zeta * mz / (2.d0 * u(:,j))
      elseif (j == (camadT - 1) .and. j == 1) then
        ws%TEupz(:,j) = ws%TEupz(:,j+1)*(exp(-u(:,camadT)*h0) + &
          RTEup(:,camadT)*FEupz(:)+RTEdw(:,camadT)*FEdwz(:)*exp(-uh(:,camadT)))
      elseif (j == (camadT - 1) .and. j /= 1) then
        ws%TEupz(:,j) = ws%TEupz(:,j+1)*(exp(u(:,camadT)*(prof(camadT-1)-h0)) + &
          RTEup(:,camadT)*FEupz(:)+RTEdw(:,camadT)*FEdwz(:) * &
          exp(-uh(:,camadT)))/(1.d0+RTEup(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j /= 1) then
        ws%TEupz(:,j) = ws%TEupz(:,j+1)*(1.d0+RTEup(:,j+1))*exp(-uh(:,j+1)) / &
          (1.d0+RTEup(:,j)*exp(-2.d0*uh(:,j)))
      elseif (j == 1) then
        ws%TEupz(:,j) = ws%TEupz(:,j+1) * (1.d0+RTEup(:,j+1))*exp(-uh(:,j+1))
      end if
    end do
  else
    ws%TEdwz(:,camadR) = zeta * mz / (2.d0 * u(:,camadT))
    ws%TEupz(:,camadR) = ws%TEdwz(:,camadR)
  end if

  twopir = 2.d0 * pi * r
  if (camadR == 1 .and. camadT /= 1) then
    fac = ws%TEupz(:,1)*exp(u(:,1)*z)
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = AdmInt(:,1) * KtezJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r
  elseif (camadR < camadT) then !camada k
    fac = ws%TEupz(:,camadR)*(exp(u(:,camadR)*(z-prof(camadR))) + &
            RTEup(:,camadR)*exp(-u(:,camadR)*(z-prof(camadR-1)+h(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (AdmInt(:,camadR) * ws%TEupz(:,camadR)*(exp(u(:,camadR)*(z-prof(camadR))) - &
      RTEup(:,camadR)*exp(-u(:,camadR)*(z-prof(camadR-1)+h(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r
  elseif (camadR == camadT .and. z <= h0) then !na mesma camada do transmissor mas acima dele
    fac = ws%TEupz(:,camadR)*(exp(u(:,camadR)*(z-h0)) + &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (AdmInt(:,camadR)*ws%TEupz(:,camadR)*(exp(u(:,camadR)*(z-h0)) - &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r
  elseif (camadR == camadT .and. z > h0) then !na mesma camada do transmissor mas abaixo dele
    fac = ws%TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-h0)) + &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (-AdmInt(:,camadR)*ws%TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-h0)) + &
      RTEup(:,camadR)*FEupz(:)*exp(-u(:,camadR)*(z-prof(camadR-1))) - &
      RTEdw(:,camadR)*FEdwz(:)*exp(u(:,camadR)*(z-prof(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r
  elseif (camadR > camadT .and. camadR /= n) then !camada j
    fac = ws%TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-prof(camadR-1))) + &
      RTEdw(:,camadR)*exp(u(:,camadR)*(z-prof(camadR)-h(camadR))))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = (-AdmInt(:,camadR)*ws%TEdwz(:,camadR)*(exp(-u(:,camadR)*(z-prof(camadR-1))) - &
      RTEdw(:,camadR)*exp(u(:,camadR)*(z-prof(camadR)-h(camadR))))) * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r
  else !camada n
    fac = ws%TEdwz(:,n)*exp(-u(:,n)*(z-prof(n-1)))
    KtezJ0 = fac * wJ0
    KtezJ1 = fac * wJ1
    KtedzzJ1 = -AdmInt(:,n) * fac * wJ1

    kernelHx = - x * sum(KtedzzJ1 * kr * kr) / twopir
    Hx_p = kernelHx / r

    kernelHy = - y * sum(KtedzzJ1 * kr * kr) / twopir
    Hy_p = kernelHy / r

    kernelHz = sum(KtezJ0 * kr * kr * kr) / 2.d0 / pi / zeta
    Hz_p = kernelHz / r
  end if
end subroutine vmd_optimized_ws
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
end module magneticdipoles
