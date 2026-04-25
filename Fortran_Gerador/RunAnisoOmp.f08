program CampoMag1DAnisotropico
! Este programa realiza as simulações de um arranjo de perfilagem Triaxial conforme especificações dadas pelo usuário, informadas
! no arquivo model.in
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! FORMATO model.in v10.0 — Multi-TR
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Layout esperado (linhas, em ordem):
!   01:          nf                         (int)
!   02..nf+1:    freq[1..nf]                (float, Hz)
!   nf+2:        ntheta                     (int)
!   nf+3..:      theta[1..ntheta]           (float, graus)
!   +1:          h1                         (altura 1º ponto-médio T-R)
!   +1:          tj                         (janela de investigação)
!   +1:          p_med                      (passo entre medidas)
!   +1:          nTR                        (int)           ← NOVO v10.0
!   +1..nTR:     dTR[1..nTR]                (float, m)      ← NOVO v10.0
!   +1:          filename                   (str)
!   +1:          ncam                       (int, nº camadas)
!   +ncam:       rho_h[i] rho_v[i]          (ncam linhas)
!   +ncam-2:     esp[1..ncam-2]             (espessuras internas)
!   +1:          modelm nmaxmodel           ("1 1" default)
!   [opcional]:  flags F5/F7/F6/filter_type (ignoradas neste executável;
!                 lidas apenas por tatu_f2py_wrapper.f08)
!
! Paridade com o lado Python em:
!   • geosteering_ai/simulation/io/model_in.py (export_model_in)
!   • tests/_fortran_helpers.py (write_model_in_multi)
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
use parameters
use DManisoTIV
implicit none
integer :: nf, ntheta, ncam, modelm, nmaxmodel, nTR !, flagdUdL
real(dp) :: h1, p_med, tj !, hn
real(dp), dimension(:), allocatable :: esp, freq, theta, dTR_arr
real(dp), dimension(:,:), allocatable :: resist
character(:), allocatable :: mypath, myfile, abspath

integer :: i, iTR

character(len=255) :: path, filename
character(len=16)  :: tr_suffix

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

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Multi-TR v10.0 — leitura de nTR seguido do array de espaçamentos T-R
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! O formato v10.0 substitui o escalar dTR (v9.0) por um bloco multi-TR:
! primeiro um inteiro nTR, em seguida nTR linhas com os valores em
! metros. Retro-compatível por construção: nTR == 1 reproduz exatamente
! o caminho single-TR do v9.0.
!
! Fail-fast: nTR < 1 é rejeitado antes de alocar — protege contra
! model.in corrompidos ou escritos com formato desconhecido.
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
read(11,*) nTR
if (nTR < 1) then
  write(*,'(a,i0)') 'Erro em model.in: nTR deve ser >= 1, lido nTR = ', nTR
  stop 1
end if
allocate(dTR_arr(nTR))
do i = 1, nTR
  read(11,*) dTR_arr(i)
end do

read(11,*) filename
! read(11,*) flagdUdL

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
read(11,*)modelm, nmaxmodel
close(11)

!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! Multi-TR v10.0 — loop sobre cada espaçamento T-R
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
! A subrotina perfila1DanisoOMP mantém a assinatura inalterada (recebe
! dTR como escalar real(dp)). O loop aqui no nível do program passa um
! valor de dTR_arr(iTR) por chamada, junto com um filename específico
! para aquele T-R.
!
! Convenção de nomes de saída:
!   nTR == 1 → {filename}.dat         (retro-compatível v9.0)
!   nTR  > 1 → {filename}_TR{i}.dat   (multi-TR v10.0)
!
! Esta convenção é consumida pelo lado Python em:
!   • tests/_fortran_helpers.py              (espera sufixo _TR{i})
!   • simulation/validation/compare_fortran.py:_locate_output_files
!     (espera {filename}.dat quando nTR == 1)
!
! Todas as camadas, arrays OpenMP caches e workspace pools são
! realocados dentro de perfila1DanisoOMP a cada iteração de TR
! (isolamento total entre TRs — sem contaminação cruzada).
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
do iTR = 1, nTR
  if (nTR == 1) then
    call perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, &
                           dTR_arr(iTR), p_med, ncam, resist, esp, trim(filename))
  else
    write(tr_suffix, '(a,i0)') '_TR', iTR
    call perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, &
                           dTR_arr(iTR), p_med, ncam, resist, esp, &
                           trim(filename)//trim(tr_suffix))
  end if
end do

deallocate(dTR_arr, freq, theta, resist, esp)

end program CampoMag1DAnisotropico
