rm(list=ls())

library(plyr)
library(tidyverse)
library(countrycode)
library(scales)
library(patchwork)
palette1 <- c("#f94144","#f3722c","#f8961e","#f9844a","#f9c74f","#90be6d","#43aa8b","#4d908e","#577590","#277da1")
palette2 <- c("#54478c","#2c699a","#048ba8","#0db39e","#16db93","#83e377","#b9e769","#efea5a","#f1c453","#f29e4c")
palette3 <- c("#f94144","#f3722c","#f8961e","#f9c74f","#90be6d","#43aa8b","#577590")
algorithms_color <- c("#f9c74f","#277da1")

# palette2 <- c("7400b8","6930c3","5e60ce","5390d9","4ea8de","48bfe3","56cfe1","64dfdf","72efdd","80ffdb")
# palette3 <- c("ff7b00","ff8800","ff9500","ffa200","ffaa00","ffb700","ffc300","ffd000","ffdd00","ffea00")
#+++++++++++++++++++++++++
# Function to calculate the mean and the standard deviation
  # for each group
#+++++++++++++++++++++++++
# data : a data frame
# varname : the name of a column containing the variable
  #to be summariezed
# groupnames : vector of column names to be used as
  # grouping variables

summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }

data_summary <- function(data, varname, groupnames){
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- plyr::rename(data_sum, c("mean" = varname))
  return(data_sum)
}

df_all <- read.csv(file = "R/sobco.csv")

algo_levels <- c("DE", "CMAES")
func_levels <- c("Sphere", "Rastrigin", "Rosenbrock", "Himmelblau")
countries_to_show <- c("USA", "ESP", "CHL", "CHN", "CAN", "IND", "GBR", "AUS", "DEU", "BRA", "NOR", "RUS", "ZAF", "MEX", "JPN", "AGO")
countries_to_show <- c("USA", "ESP", "CHN", "IND", "GBR", "DEU", "CHL")
dim_levels <- paste0("D = ", seq(10,130,40))

df_all <- df_all %>%
  mutate_if(is.character, as.factor) %>%
  mutate(algorithm = str_to_upper(algorithm)) %>%
  mutate(algorithm=factor(algorithm, levels = algo_levels)) %>%
  dplyr::rename("Algorithm" = algorithm) %>%
  mutate(function. = str_to_title(function.)) %>%
  mutate(function.=factor(function., levels = func_levels)) %>%
  dplyr::rename("OF" = function.) %>%
  mutate(D = paste("D", dimension, sep = " = ")) %>%
  mutate(D = factor(D, levels = dim_levels)) %>%
  dplyr::filter(OF %in% func_levels[1:3]) %>%
  dplyr::filter(country_code %in% countries_to_show)

df_all_run <- df_all %>% dplyr::filter(country_code == "USA")

df_summ <- data_summary(df_all_run, varname="kg_carbon",
                    groupnames=c("OF", "D", "Algorithm"))

# df_all_g <- df_all %>% gather(value = "social_cost", key = "level", -1:-13)

ggplot(df_summ, aes(x=Algorithm, y=kg_carbon, fill=Algorithm)) +
  geom_bar(stat="identity",
           position=position_dodge(), width = 0.5) +
  geom_errorbar(aes(ymin=kg_carbon-sd, ymax=kg_carbon+sd), width=.2,
                 position=position_dodge(.9)) +
  labs(title="KgCO2 per run", y="KgCO2/run", x = "Algorithm")+
   theme_minimal() +
   scale_fill_manual(values=algorithms_color) +
  facet_grid(facets = D ~ OF, scales = "free_y") +
  theme(legend.position = "none")

df_summ <- data_summary(df_all_run, varname="total_power",
                    groupnames=c("OF", "D", "Algorithm"))

ggplot(df_summ, aes(x=Algorithm, y=total_power, fill=Algorithm)) +
  geom_bar(stat="identity",
           position=position_dodge(), width = 0.5) +
  geom_errorbar(aes(ymin=total_power-sd, ymax=total_power+sd), width=.2,
                 position=position_dodge(.9)) +
  labs(title="Energy per run", y="kWh", x = "Algorithm")+
   theme_minimal() +
   scale_fill_manual(values=algorithms_color) +
  facet_grid(facets = D ~ OF, scales = "free_y") +
  theme(legend.position = "none")


df_total <- df_all_run %>%
  group_by(OF,D,dimension,Algorithm) %>%
  dplyr::summarise(kg_carbon = sum(kg_carbon), total_power = sum(total_power)) %>%
  mutate(vjust_ = ifelse(total_power < 0.10, -0.5, -0.5))

ggplot(df_total, aes(x=Algorithm, y=total_power, fill=Algorithm)) +
  geom_bar(stat="identity",
           position=position_dodge(), width = 0.5, color="black") +
  scale_y_continuous(limits = c(0, max(df_total$total_power)+0.005)) +
  labs(title="Total energy consumption of SOBCO experiments (kWh)", y="kWh", x = "Metaheuristic")+
  geom_text(aes(label = format(round(total_power, 4), nsmall = 4), vjust=vjust_), hjust=0.5, size=2.5, fontface=2) +
   scale_fill_manual(values=algorithms_color) +
  facet_grid(facets = OF ~ D) +
  theme(legend.position = "none",
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

ggsave("total_energy_consumption_sobco.pdf", width = 7, height = 3.375)


ggplot(df_total, aes(x=Algorithm, y=kg_carbon, fill=Algorithm)) +
  geom_bar(stat="identity",
           position=position_dodge(), width = 0.5, color="black") +
  scale_y_continuous(limits = c(0, max(df_total$kg_carbon)+0.015)) +
  labs(title="Total of carbon emissions of SOBCO experiments (KgCO2)", y="Kg CO2", x = "Metaheuristic")+
  geom_text(aes(label = format(round(kg_carbon, 4), nsmall = 4), vjust=vjust_), hjust=0.5, size=2.5, fontface=2) +
   scale_fill_manual(values=algorithms_color) +
  facet_grid(facets = OF ~ D) +
  theme(legend.position = "none",
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

ggsave("total_carbon_emissions_sobco.pdf", width = 7, height = 3.375)

scientific_formatter <-  function(x) {
  #parse(text=gsub("e\\+*", " %*% 10^", scales::scientific_format(digits = 3)(x)))
  scales::scientific_format(digits = 1)(x)
}

ggplot(df_all_run, aes(x=Algorithm, y=best_fitness, fill=Algorithm)) +
  geom_boxplot(width = 0.5) +
  scale_y_continuous(labels = scientific_formatter) +
  labs(title="Algorithm performance in the SOBCO scenario (30 runs)", y="Best fitness", x = "Metaheuristic")+
  scale_fill_manual(values=algorithms_color) +
  facet_wrap(facets = OF ~ D, scales = "free") +
  theme(legend.position = "none")

ggsave("performance_sobco.pdf", width = 8, height = 5)


ggplot(df_all, aes(x=best_fitness, y=kg_carbon)) +
  geom_point(aes(color=Algorithm, shape=Algorithm), size=1) +
  labs(title="Emissions vs. performance", y="Emissions (Kg/run)", x = "Best fitness (run)") +
  scale_color_manual(values=algorithms_color) +
  theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust=1)) +
  facet_grid(facets = OF ~ D, scales = "free")

ggsave("performance_emissions.pdf", width = 7, height = 3.375)

ggplot(df_total, aes(x=dimension, y=total_power, colour = Algorithm, shape=Algorithm)) +
  geom_line() +
  geom_point(aes(colour = Algorithm), size = 3) +
  geom_point(colour = "white", size = 1.5) +
  scale_color_manual(values=algorithms_color) +
  scale_x_continuous(breaks = seq(10,130,40), limits = c(5, 135)) +
  #scale_y_continuous(limits = c(0, 1.8)) +
  ylab("kWh") +
  facet_grid(OF~.)



df_social_cost_countries <- df_all %>% select(country_code, ends_with("cost")) %>%
  #gather(value = "social_cost", key = "level", -1) %>%
  group_by(country_code) %>%
  dplyr::summarise(
    median_carbon_cost = sum(median_carbon_cost),
    lower_carbon_cost = sum(lower_carbon_cost),
    upper_carbon_cost = sum(upper_carbon_cost)
  )

df_social_cost_countries$country_code <- countrycode(df_social_cost_countries$country_code, "iso3c", "country.name")

arr_countries <- df_social_cost_countries %>% arrange(median_carbon_cost)
df_social_cost_countries$country_code <- factor(
  df_social_cost_countries$country_code,
  levels = arr_countries$country_code
)



ggplot(df_social_cost_countries,
       aes(x=country_code, y=median_carbon_cost, fill = country_code)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin=median_carbon_cost-lower_carbon_cost,
                    ymax=median_carbon_cost+upper_carbon_cost), width=0.5) +
  ylab("Dollars ($)") +
  xlab("Country") +
  ggtitle("Social cost (dollars per tonne of CO2) by country") +
  theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust=1)) +
  theme(legend.position = "none")

ggsave("social_cost_by_country.pdf", width = 7, height = 4)

ggplot(df_social_cost_countries,
       aes(x=country_code, y=median_carbon_cost, color = country_code)) +
  geom_errorbar(aes(ymin=median_carbon_cost-lower_carbon_cost,
                    ymax=median_carbon_cost+upper_carbon_cost), width=0.5) +
  geom_point(size = 2) +
  geom_point(color="white", size = 1) +
  ylab("Dollars ($)") +
  xlab("Country") +
  ggtitle("Social cost (dollars per tonne of CO2) by country") +
  theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust=1)) +
  theme(legend.position = "none")

ggsave("social_cost_by_country_2.pdf", width = 7, height = 4)



# Dynamic experiment

peaks_levels <- paste0("Peaks = ", seq(10,40,10))
change_freq_levels <- paste0("CF = ", seq(2500,10000,2500))
swarms_levels <- paste0("Swarms = ", c(1,10,20,30))


df_all_dyn <- read.csv(file = "R/edo.csv")

df_all_dyn <- df_all_dyn %>%
  mutate_if(is.character, as.factor) %>%
  dplyr::rename("Algorithm" = algorithm) %>%
  dplyr::filter(country_code %in% countries_to_show)

df_all_run_dyn <- df_all_dyn %>% dplyr::filter(country_code == "USA")

df_order <- read.delim("R/dynamic_order.txt", sep = ",")

df_all_complete_run <- df_all_run_dyn %>% dplyr::bind_cols(df_order %>% select(-run)) %>% mutate(PeakFunction="cone")

df_all_complete_run <- df_all_complete_run %>%
    mutate(Swarms = paste("Swarms", n_swarms, sep = " = ")) %>%
    mutate(Swarms = factor(Swarms, levels = swarms_levels)) %>%
    mutate(CF = paste("CF", change_freq, sep = " = ")) %>%
    mutate(CF = factor(CF, levels = change_freq_levels)) %>%
    mutate(Peaks = paste("Peaks", n_peaks, sep = " = ")) %>%
    mutate(Peaks = factor(Peaks, levels = peaks_levels))

df_total_dyn <- df_all_complete_run %>%
  group_by(CF,n_swarms,Swarms,Peaks) %>%
  dplyr::summarise(kg_carbon = sum(kg_carbon), total_power = sum(total_power)) %>%
  mutate(vjust_ = ifelse(total_power < 0.10, -0.5, -0.5)) %>% ungroup()

ggplot(df_total_dyn, aes(x=as.factor(n_swarms), y=kg_carbon, fill=Swarms)) +
  geom_bar(stat="identity",
           position=position_dodge(), width = 0.9, color="black") +
  scale_y_continuous(limits = c(0, max(df_total_dyn$kg_carbon)+0.0015)) +
  labs(title="Total of carbon emissions of EDO experiments (KgCO2)", y="Kg CO2", x = "Swarms")+
  geom_text(aes(label = formatC(kg_carbon, format = "f", digits = 4), vjust=vjust_), hjust=0.5, size=2.5, fontface=2) +
  scale_fill_manual(values=c("#f3722c","#f9c74f","#4d908e","#277da1")) +
  facet_grid(facets = Peaks ~ CF) +
  theme_grey() +
  theme(legend.position = "none",
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

ggsave("total_carbon_emissions_edo.pdf", width = 7, height = 4.5)

ggplot(df_total_dyn, aes(x=as.factor(n_swarms), y=total_power, fill=Swarms)) +
  geom_bar(stat="identity",
           position=position_dodge(), width = 0.9, color="black") +
  scale_y_continuous(limits = c(0, max(df_total_dyn$total_power)+0.001)) +
  labs(title="Total energy consumption of EDO experiments (kWh)", x = "Swarms")+
  geom_text(aes(label = formatC(total_power, format = "f", digits = 4), vjust=vjust_), hjust=0.5, size=2.5, fontface=2) +
  scale_fill_manual(values=c("#f3722c","#f9c74f","#4d908e","#277da1")) +
  facet_grid(facets = Peaks ~ CF) +
  theme_grey() +
  theme(legend.position = "none",
        axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

ggsave("total_energy_consumption_edo.pdf", width = 7, height = 4.5)


# Performance

ggplot(df_all_complete_run, aes(x=as.factor(n_swarms), y=mean_offline_error, fill=Swarms)) +
  geom_boxplot(width = 0.7) +
  scale_y_continuous(labels = scientific_formatter) +
  labs(title="Algorithm performance in the EDO scenario (30 runs)", y="Offline error", x = "Swarms")+
  scale_fill_manual(values=c("#f3722c","#f9c74f","#4d908e","#277da1")) +
  facet_wrap(facets = Peaks ~ CF, scales = "free") +
  theme(legend.position = "none")

ggsave("performance_edo.pdf", width = 8, height = 7)



# Social cost

df_social_cost_countries_dyn <- df_all_dyn %>% select(country_code, ends_with("cost")) %>%
  #gather(value = "social_cost", key = "level", -1) %>%
  group_by(country_code) %>%
  dplyr::summarise(
    median_carbon_cost = sum(median_carbon_cost),
    lower_carbon_cost = sum(lower_carbon_cost),
    upper_carbon_cost = sum(upper_carbon_cost)
  )

df_social_cost_countries_dyn$country_code <- countrycode(df_social_cost_countries_dyn$country_code, "iso3c", "country.name")

arr_countries <- df_social_cost_countries_dyn %>% arrange(median_carbon_cost)
df_social_cost_countries_dyn$country_code <- factor(
  df_social_cost_countries_dyn$country_code,
  levels = arr_countries$country_code
)



ggplot(df_social_cost_countries_dyn,
       aes(x=country_code, y=median_carbon_cost, fill = country_code)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin=median_carbon_cost-lower_carbon_cost,
                    ymax=median_carbon_cost+upper_carbon_cost), width=0.5) +
  ylab("Dollars ($)") +
  xlab("Country") +
  ggtitle("Social cost (dollars per tonne of CO2) by country") +
  theme(axis.text.x = element_text(angle = 30, vjust = 1, hjust=1)) +
  theme(legend.position = "none")

ggsave("social_cost_by_country_edo.pdf", width = 7, height = 4)

df_social_cost_countries$Scenario <- "SOBCO"
df_social_cost_countries_dyn$Scenario <- "EDO"

df_both <- df_social_cost_countries %>% bind_rows(df_social_cost_countries_dyn) %>%
  mutate(Scenario = factor(Scenario, levels = c("SOBCO", "EDO")))

p4 <- ggplot(df_both,
       aes(x=country_code, y=median_carbon_cost, fill = Scenario)) +
  geom_bar(stat = "identity", width = 0.8, position = position_dodge(width=0.8), color="black") +
  scale_y_continuous(breaks = seq(-0.01,0.06,0.01)) +
  geom_errorbar(aes(ymin=median_carbon_cost-lower_carbon_cost,
                    ymax=median_carbon_cost+upper_carbon_cost,
                    group=Scenario),
                position = position_dodge(width=0.8),
                # linetype="dashed",
                width=0.5) +
  ylab("Dollars per tonne of CO2") +
  xlab("Country") +
  scale_fill_manual(values=algorithms_color) +
  ggtitle("d) Social cost by country") +
  theme(legend.position=c(.1,.8))

ggsave("social_cost_scenarios.pdf", plot = p4, width = 7, height = 3.5)


# Summary of both scenarios
scenario_levels <- c("SOBCO", "EDO")
df_time <- data.frame(
  Scenario = scenario_levels,
  TimeInHours = c(17.52573503861111, 20.085046700555555),
  Label = c('17:31:32', '20:05:06')
)

df_time$Scenario <- factor(df_time$Scenario, levels = scenario_levels)

p1 <- ggplot(df_time, aes(x = Scenario, y = TimeInHours, fill = Scenario)) +
  geom_bar(stat = "identity", width = 0.5, color="black") +
  labs(title="a) Execution time", x = "Scenario", y = "Hours") +
  geom_text(aes(label = Label), vjust=-0.5, hjust=0.5, size=2.5, fontface=2) +
  scale_y_continuous(limits = c(0, max(df_time$TimeInHours)+2)) +
  theme(legend.position = "none") +
  scale_fill_manual(values=algorithms_color)

df_total_sobco_comp <- df_total %>% ungroup() %>%
  summarise(total_power = sum(total_power),
            kg_carbon = sum(kg_carbon)) %>%
  mutate(Scenario = "SOBCO")

df_total_edo_comp <- df_total_dyn %>% ungroup() %>%
  summarise(total_power = sum(total_power),
            kg_carbon = sum(kg_carbon)) %>%
  mutate(Scenario = "EDO")

df_comp <- df_total_sobco_comp %>% bind_rows(df_total_edo_comp)

df_comp$Scenario <- factor(df_comp$Scenario, levels = scenario_levels)

p2 <- ggplot(df_comp, aes(x = Scenario, y = total_power, fill = Scenario)) +
  geom_bar(stat = "identity", width = 0.5, color="black") +
  labs(title="b) Energy", x = "Scenario", y = "kWh") +
  geom_text(aes(label = formatC(total_power, format = "f", digits = 3)), vjust=-0.5, hjust=0.5, size=2.5, fontface=2) +
  scale_y_continuous(limits = c(0, max(df_comp$total_power)+0.025)) +
  theme(legend.position = "none") +
  scale_fill_manual(values=algorithms_color)

p3 <- ggplot(df_comp, aes(x = Scenario, y = kg_carbon, fill = Scenario)) +
  geom_bar(stat = "identity", width = 0.5, color="black") +
  labs(title="c) CO2 emissions", x = "Scenario", y = "KgCO2") +
  geom_text(aes(label = formatC(kg_carbon, format = "f", digits = 3)), vjust=-0.5, hjust=0.5, size=2.5, fontface=2) +
  scale_y_continuous(limits = c(0, max(df_comp$kg_carbon)+0.05)) +
  theme(legend.position = "none") +
  scale_fill_manual(values=algorithms_color)





df_both <- df_both %>% mutate(info =
                              paste0(
                                formatC(median_carbon_cost, format = "f", digits = 3),
                                " (",
                                formatC(lower_carbon_cost, format = "f", digits = 3),
                                ", ",
                                formatC(upper_carbon_cost, format = "f", digits = 3),
                                ")"
                              )
) %>% mutate(just_ = ifelse(median_carbon_cost > 0.0082, 1.05, ifelse(median_carbon_cost < -0.0001, -0.15, -0.05)))

p5 <- ggplot(df_both,
       aes(x=country_code, y=median_carbon_cost, fill = Scenario)) +
  geom_bar(stat = "identity", width = 0.8, position = position_dodge(width=0.8), color="black") +
  scale_y_continuous(breaks = seq(-0.01,0.06,0.01)) +
  #geom_errorbar(aes(ymin=median_carbon_cost-lower_carbon_cost,
  #                  ymax=median_carbon_cost+upper_carbon_cost,
  #                  group=Scenario),
  #              position = position_dodge(width=0.8),
                # linetype="dashed",
  #              width=0.5) +
  geom_text(aes(label = info, hjust=just_), angle=90, size = 2.5, position = position_dodge(width=0.8), fontface=2) +
  ylab("Dollars per tonne of CO2") +
  xlab("Country") +
  scale_fill_manual(values=algorithms_color) +
  ggtitle("d) Social cost by country") +
  theme(legend.position=c(.15,.8)) +
  guides(fill = guide_legend(nrow = 1))


p <- (p1 + p2 + p3) / p5 + plot_layout(heights = c(1, 2))

ggsave("scenarios_comparison.pdf", plot = p, width = 7, height = 5)

df_final <- df_both %>% ungroup() %>% group_by(country_code) %>% summarise(
  median_cost = sum(median_carbon_cost),
  lower_cost = sum(lower_carbon_cost),
  upper_cost = sum(upper_carbon_cost),
)

df_final_energy_emissions_sobco <- df_all_run %>% ungroup() %>% summarise(
  total_power = sum(total_power),
  kg_carbon = sum(kg_carbon),
)

df_final_energy_emissions_edo <- df_all_complete_run %>% ungroup() %>% summarise(
  total_power = sum(total_power),
  kg_carbon = sum(kg_carbon),
)

df_final_energy_emissions <- df_final_energy_emissions_sobco %>%
  bind_rows(df_final_energy_emissions_edo) %>%
  summarise(
  total_power = sum(total_power),
  kg_carbon = sum(kg_carbon),
)
