library(RNetCDF)
library(maps)
library(sp)
library(fields)
setwd("/Users/xx249/Documents/ProjectsExe/df_revision/elevCodes/")

elev_data = load("elev.RData")

analysis.files <- list.files(pattern="analysis")
obs.files <- list.files(pattern="obs")

analysis.nc <- open.nc(analysis.files[1])
obs.nc <- open.nc(obs.files[1])

analysis.footprint <- list()
analysis.footprint$lon <- var.get.nc(analysis.nc, 2) - 360
analysis.footprint$lat <- var.get.nc(analysis.nc, 3)
analysis.footprint.z <- var.get.nc(analysis.nc, 5)
xy <- as.matrix(expand.grid(analysis.footprint))

mp <- map("world", "france", fill=TRUE, plot=FALSE)

rotateCoords <- function(coords, pole) {
if (length(coords) == 2) coords <- matrix(coords, 1)
degtorad <- pi/180
pole.long <- (pole[1] %% 360) * degtorad
pole.latt <- (pole[2] %% 360) * degtorad
SOCK <- pole.long - pi
if (pole.long==0) SOCK <- 0
longit <- (coords[,1] %% 360) * degtorad
latt <- (coords[,2] %% 360) * degtorad
SCREW <- longit - SOCK
SCREW <- SCREW %% (2*pi)
BPART <- cos(SCREW) * cos(latt)
rlatt <- asin(-cos(pole.latt) * BPART + sin(pole.latt)*sin(latt))
t1 <- cos(pole.latt)*sin(latt)
t2 <- sin(pole.latt) * BPART
rlong <- -acos((t1 + t2)/cos(rlatt))
id <- 0 < SCREW & SCREW < pi | SCREW > 2*pi
rlong[id] <- -rlong[id]
180*cbind(rlong, rlatt)/pi
}

id <- character(length(mp$x))
mp.na <- which(is.na(mp$x))
starts <- c(1, mp.na + 1)
ends <- c(mp.na - 1, length(mp$x))
for (i in seq_along(mp$names)) id[starts[i]:ends[i]] <- mp$names[i]
france.coords <- cbind(mp$x[which(id == "France")], mp$y[which(id == "France")])
france.rcoords <- rotateCoords(france.coords, pole=c(193, 41))

is.france <- as.logical(point.in.polygon(xy[,1], xy[,2], france.rcoords[,1], france.rcoords[,2]))
dataMoFr = cbind(xy[is.france,], elev_fp[is.france], analysis.footprint.z[is.france])
save(dataMoFr,  file = 'my_dataMo_FR_elev.RData')
plot(xy[is.france,])

obs.lon <- var.get.nc(obs.nc, 3)
obs.lat <- var.get.nc(obs.nc, 2)
obs.alt <- var.get.nc(obs.nc, 6)
obs.coords <- cbind(obs.lon, obs.lat)
obs.coords.rotated <- rotateCoords(obs.coords, pole=c(193, 41))
obs <- var.get.nc(obs.nc, 7)
dataObs = cbind(obs.coords, elev_obs, obs)
save(dataObs,  file = 'my_dataObs_elev.RData')

x= obs.lon
y = obs.lat
country <- map.where("world", x, y)
id.france <- country == "France"
id.france[is.na(id.france)] <- FALSE # this is because map.where generates NA for non-land points

# a quick test plot
# plot all points
plot(x, y, pch=20)
# plot in red the France points
points(x[id.france], y[id.france], col="red", pch=20)
# check it's right
map("world", add=TRUE)

obs.coords.fr <- cbind(x[id.france], y[id.france])
obs.coords.rotated.fr <- rotateCoords(obs.coords.fr, pole=c(193, 41))
obs.fr = obs[id.france]
#dataObsFr = cbind(obs.coords.rotated.fr, obs.fr)
#save(dataObsFr,  file = 'my_dataObs_FR.RData')

obs.alt = elev_obs[id.france]
dataObsFr = cbind(obs.coords.rotated.fr, obs.alt, obs.fr)
save(dataObsFr,  file = 'my_dataObs_FR_elev.RData')

meanObsFr =  mean(obs.fr)
obs.fr = obs.fr - meanObsFr

library(akima)
x= dataMoFr[,1]
y= dataMoFr[,2]
z= elev_fp
x.plot <-  seq(-11.7, -3.21, l=500)
y.plot <- seq(-6.2, 3.0, l=500)

int <- interp(x, y, elev_fp[is.france], xo=x.plot, yo=y.plot)

plot.seq <- pretty(-3:3355, 20)
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
pal <- jet.colors(length(plot.seq) - 1)
image.plot(int, breaks=plot.seq, col=pal)

col.pts <- pal[as.numeric(cut(obs.fr, plot.seq))]
points(dataObsFr[,1], dataObsFr[,2], bg=col.pts, pch=21)

library(akima)
x= dataMoFr[,1]
y= dataMoFr[,2]
z= dataMoFr[,3] - meanObsFr
x.plot <-  seq(-11.7, -3.21, l=500)
y.plot <- seq(-6.2, 3.0, l=500)

int <- interp(x, y, z, xo=x.plot, yo=y.plot)

plot.seq <- pretty(-17:17, 20)
jet.colors <- colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
pal <- jet.colors(length(plot.seq) - 1)
image.plot(int, breaks=plot.seq, col=pal)

col.pts <- pal[as.numeric(cut(obs.fr, plot.seq))]
points(dataObsFr[,1], dataObsFr[,2], bg=col.pts, pch=21)

