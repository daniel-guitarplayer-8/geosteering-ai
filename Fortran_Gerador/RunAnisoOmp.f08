program CampoMag1DAnisotropico
! Este programa realiza as simulações de um arranjo de perfilagem Triaxial conforme
! especificações dadas pelo usuário, informadas no arquivo model.in.
!
! Versão 9.0 — suporte a:
!   - Feature 1: múltiplos pares T-R (nTR espaçamentos) [v7.0]
!   - F5: frequências arbitrárias (nf > 2, guard + validação) [v8.0]
!   - F6: compensação midpoint (borehole compensation) [v9.0]
!   - F7: antenas inclinadas (tilted coils, pós-processamento do tensor) [v8.0]
!   - Filtro Adaptativo: seleção Kong/Werthmuller/Anderson [v9.0]
!
! Formato do model.in (v9.0):
!   nf                              ! número de frequências
!   freq(1) ... freq(nf)            ! frequências em Hz (1 por linha)
!   ntheta                          ! número de ângulos
!   theta(1) ... theta(ntheta)      ! ângulos em graus (1 por linha)
!   h1                              ! profundidade inicial (m)
!   tj                              ! janela de investigação (m)
!   p_med                           ! passo de medição (m)
!   nTR                             ! número de pares T-R
!   dTR(1) ... dTR(nTR)             ! espaçamentos T-R (m, 1 por linha)
!   filename                        ! nome base do arquivo de saída
!   ncam                            ! número de camadas geológicas
!   resist(1,1) resist(1,2)         ! resistividades h/v por camada
!   ...
!   resist(ncam,1) resist(ncam,2)
!   esp(2) ... esp(ncam-1)          ! espessuras internas (1 por linha)
!   modelm nmaxmodel                ! índice do modelo e total
!   --- Seção opcional v9.0 (backward compatible via iostat) ---
!   use_arbitrary_freq              ! F5: 0=desabilitado (default), 1=habilitado
!   use_tilted_antennas             ! F7: 0=desabilitado (default), 1=habilitado
!   n_tilted                        ! (só se F7=1) número de configs tilted
!   beta(1) phi(1)                  ! (só se F7=1) inclinação e azimute em graus
!   ...
!   beta(n_tilted) phi(n_tilted)
!   use_compensation                ! F6: 0=desabilitado (default), 1=habilitado
!   n_comp_pairs                    ! (só se F6=1) número de pares de compensação
!   near(1) far(1)                  ! (só se F6=1) índices dos pares T-R
!   ...
!   near(n_comp_pairs) far(n_comp_pairs)
!   filter_type                     ! 0=Werthmuller (default), 1=Kong, 2=Anderson
!
use parameters
use DManisoTIV
implicit none
integer :: nf, ntheta, ncam, modelm, nmaxmodel, nTR
real(dp) :: h1, p_med, tj
real(dp), dimension(:), allocatable :: esp, freq, theta, dTR_arr
real(dp), dimension(:,:), allocatable :: resist
character(:), allocatable :: mypath, myfile, abspath

integer :: i

character(len=255) :: path, filename

! F5/F7 — Variáveis para features opcionais
integer :: use_arb_freq, use_tilted, n_tilted, ios
real(dp), dimension(:), allocatable :: beta_tilt, phi_tilt
! F6 — Variáveis para compensação midpoint
integer :: use_compensation, n_comp_pairs, n_pairs_to_skip
integer, dimension(:,:), allocatable :: comp_pairs
! Filtro Adaptativo — tipo de filtro de Hankel
integer :: filter_type

call getcwd(path)
mypath = trim(path)//'/'
myfile = 'model.in'
abspath = mypath//myfile

open( unit = 11, file = abspath, status = 'old', action = 'read' )
read(11,*) nf
allocate(freq(nf))
do i = 1, nf
  read(11,*)freq(i)
end do
read(11,*) ntheta
allocate(theta(ntheta))
do i = 1, ntheta
  read(11,*)theta(i)
end do
read(11,*) h1
read(11,*) tj
read(11,*) p_med

! Leitura de múltiplos pares T-R (Feature 1):
! model.in especifica nTR (número de pares T-R) seguido de nTR valores de dTR.
! Exemplo nTR=1: "1\n1.0"  Exemplo nTR=3: "3\n0.5\n1.0\n2.0"
read(11,*) nTR
allocate(dTR_arr(nTR))
do i = 1, nTR
  read(11,*) dTR_arr(i)
end do

read(11,*) filename

read(11,*) ncam
allocate(resist(ncam,2), esp(ncam))
do i = 1,ncam
    read(11,*) resist(i,1), resist(i,2)
end do
esp(1) = 0.d0
do i = 2,ncam-1
    read(11,*) esp(i)
end do
esp(ncam) = 0.d0
read(11,*) modelm, nmaxmodel

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! F5/F7 — Leitura de flags opcionais (backward compatible)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Se o model.in terminar após modelm/nmaxmodel (formato v7.0 ou anterior),
! o read com iostat retorna EOF e os defaults (0 = desabilitado) são usados.
! Isso garante que model.in antigos continuem funcionando sem alteração.
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
use_arb_freq     = 0
use_tilted       = 0
n_tilted         = 0
use_compensation = 0
n_comp_pairs     = 0
filter_type      = 0

! F5 — Leitura do flag de frequências arbitrárias
read(11, *, iostat=ios) use_arb_freq
if (ios /= 0) then
  ! EOF ou erro: manter defaults (v7.0 backward compat)
  use_arb_freq = 0
  close(11)
  allocate(beta_tilt(0), phi_tilt(0))
  allocate(comp_pairs(0,2))
  goto 900
end if

! F7 — Leitura do flag de antenas inclinadas
read(11, *, iostat=ios) use_tilted
if (ios /= 0) then
  use_tilted = 0
  close(11)
  allocate(beta_tilt(0), phi_tilt(0))
  allocate(comp_pairs(0,2))
  goto 900
end if

! F7 — Leitura dos parâmetros de antenas inclinadas (se habilitado)
if (use_tilted == 1) then
  read(11, *, iostat=ios) n_tilted
  if (ios /= 0 .or. n_tilted < 1) then
    write(*,'(A)') '[F7 AVISO] use_tilted=1 mas n_tilted inválido. Desabilitando F7.'
    use_tilted = 0
    n_tilted = 0
    allocate(beta_tilt(0), phi_tilt(0))
    close(11)
    allocate(comp_pairs(0,2))
    goto 900
  else
    allocate(beta_tilt(n_tilted), phi_tilt(n_tilted))
    do i = 1, n_tilted
      read(11, *, iostat=ios) beta_tilt(i), phi_tilt(i)
      if (ios /= 0) then
        write(*,'(A,I0)') '[F7 ERRO] Falha ao ler configuração tilted #', i
        stop '[F7] Dados de antenas inclinadas incompletos no model.in'
      end if
    end do
  end if
else
  allocate(beta_tilt(0), phi_tilt(0))
end if

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! F6 — Leitura do flag de compensação midpoint (v9.0)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Se o model.in terminar após F7, o read com iostat retorna EOF e o default
! (0 = desabilitado) é mantido. Backward compatible com model.in v8.0.
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
read(11, *, iostat=ios) use_compensation
if (ios /= 0) then
  use_compensation = 0
  allocate(comp_pairs(0,2))
  close(11)
  goto 900
end if

! F6 — Leitura dos pares de compensação (se habilitado)
if (use_compensation == 1) then
  read(11, *, iostat=ios) n_comp_pairs
  if (ios /= 0 .or. n_comp_pairs < 1) then
    write(*,'(A)') '[F6 AVISO] use_compensation=1 mas n_comp_pairs inválido. Desabilitando F6.'
    use_compensation = 0
    n_comp_pairs = 0
    allocate(comp_pairs(0,2))
    close(11)
    goto 900
  else
    if (nTR < 2) then
      write(*,'(A)') '[F6 AVISO] use_compensation=1 mas nTR < 2. Desabilitando F6.'
      ! Salvar n_comp_pairs antes de zerar para consumir as linhas do model.in
      n_pairs_to_skip = n_comp_pairs
      use_compensation = 0
      n_comp_pairs = 0
      allocate(comp_pairs(0,2))
      ! Consumir linhas restantes de F6 para não desalinhar leitura do filter_type
      do i = 1, n_pairs_to_skip
        read(11, *, iostat=ios)
      end do
    else
      allocate(comp_pairs(n_comp_pairs, 2))
      do i = 1, n_comp_pairs
        read(11, *, iostat=ios) comp_pairs(i, 1), comp_pairs(i, 2)
        if (ios /= 0) then
          write(*,'(A,I0)') '[F6 ERRO] Falha ao ler par de compensação #', i
          stop '[F6] Dados de compensação incompletos no model.in'
        end if
        ! Validação: índices devem estar em [1, nTR]
        if (comp_pairs(i,1) < 1 .or. comp_pairs(i,1) > nTR .or. &
            comp_pairs(i,2) < 1 .or. comp_pairs(i,2) > nTR) then
          write(*,'(A,I0,A,I0,A,I0)') &
            '[F6 ERRO] Par #', i, ': near=', comp_pairs(i,1), ' far=', comp_pairs(i,2)
          stop '[F6] Índices de compensação fora do intervalo [1, nTR]'
        end if
      end do
    end if
  end if
else
  allocate(comp_pairs(0,2))
end if

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Filtro Adaptativo — Leitura do tipo de filtro de Hankel (v9.0)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! 0 = Werthmuller 201pt (default, backward compatible)
! 1 = Kong 61pt (rápido, para geração de datasets de treinamento)
! 2 = Anderson 801pt (máxima precisão, para validação)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
read(11, *, iostat=ios) filter_type
if (ios /= 0) then
  filter_type = 0  ! Default: Werthmuller
end if
! Validação: filter_type deve estar em [0, 2]
if (filter_type < 0 .or. filter_type > 2) then
  write(*,'(A,I0,A)') '[FILTRO AVISO] filter_type=', filter_type, ' inválido. Usando Werthmuller (0).'
  filter_type = 0
end if

close(11)

900 continue

call perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, &
                       nTR, dTR_arr, p_med, ncam, resist, esp, filename, &
                       use_arb_freq, use_tilted, n_tilted, beta_tilt, phi_tilt, &
                       use_compensation, n_comp_pairs, comp_pairs, &
                       filter_type)

end program CampoMag1DAnisotropico
