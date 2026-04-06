program CampoMag1DAnisotropico
! Este programa realiza as simulações de um arranjo de perfilagem Triaxial conforme especificações dadas pelo usuário, informadas
! no arquivo model.in
!
! Versão 7.0 — suporte a múltiplos pares T-R (nTR espaçamentos)
! O model.in agora especifica nTR (inteiro) seguido de nTR valores de dTR.
! Para backward-compatibility, nTR=1 produz saída idêntica ao formato anterior.
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
read(11,*)modelm, nmaxmodel
close(11)
call perfila1DanisoOMP(modelm, nmaxmodel, mypath, nf, freq, ntheta, theta, h1, tj, &
                       nTR, dTR_arr, p_med, ncam, resist, esp, filename)

end program CampoMag1DAnisotropico
