from setuptools import setup

package_name = 'qcar2_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ziyad',
    maintainer_email='ziyad@todo.todo',
    description='QCar2 simple localization node',
    license='TODO',
    entry_points={
        'console_scripts': [
            'localization_node = qcar2_localization.localization_node:main',
        ],
    },
)